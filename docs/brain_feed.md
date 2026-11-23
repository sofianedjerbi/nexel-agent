# The brain feed

What an AI player sees, thinks, decides and remembers, as one small JSON object per line (JSON Lines). The live body
writes it (`rl/brain.py`, `python -m nexel_agent.rl.live ... --brain 8790`): appended to a file and served as
Server-Sent Events on `http://127.0.0.1:8790/feed`. The brain view (`tools/brain/index.html`) reads either:

    tools/brain/index.html?feed=http://127.0.0.1:8790/feed      live
    tools/brain/index.html?replay=demo_replay.jsonl              a recording, with play/pause, 1-8x and a scrubber

Rules that keep it stable:

- Every line has `k` (the kind) and `t` (Unix seconds, one decimal). Lines come in time order.
- A field that is not known is left out, never sent as a guess. The page hides what is missing.
- New fields may be added; a field never changes meaning. `hello.v` goes up only if one did.
- Names of people are the game's names. No file paths, no e-mails, no keys.

## Kinds

### `hello`: once, first

```json
{"k": "hello", "t": 1795460353.0, "v": 1, "name": "Rook", "persona": "a quiet player who learned the world by living in it"}
```

### `frame`: what it sees and does, about once a second

```json
{"k": "frame", "t": 1795460440.2, "pos": [7, -35, 2], "hp": 17, "food": 16, "facing": "W", "time": "day 836, 18:44 (night)",
 "tiles": ["..T..........", "...", "... 13 rows of 13"],
 "beings": [{"name": "doben00", "kind": "agent", "dx": 2, "dy": -1, "hp": 14, "side": "ally"},
            {"kind": "sporeling", "dx": -3, "dy": 0, "hp": 2, "side": "foe"}],
 "inv": {"stone_block": 21, "dirt": 2, "path": 2},
 "goal": {"kind": "gather", "item": "wood", "with": "doben00", "since_s": 86, "have": 0},
 "top": [["dig_W", 0.41], ["step_W", 0.22], ["attack", 0.12], ["say", 0.05], ["step_N", 0.04], ["eat", 0.03]],
 "act": "dig_W", "legal": 37, "of": 184, "value": 0.42}
```

| field | meaning |
|---|---|
| `pos` | world tile `[x, y]`, and `z` (its height) when known |
| `hp`, `food` | out of 20 (food: the game's `hunger`, 20 = full) |
| `facing` | `N`, `E`, `S` or `W`: the way its last step, jump, dig or climb went |
| `time` | the game clock as the game says it |
| `tiles` | the 13 x 13 tiles round it, the middle of the agent API's grid: row 0 is north, `tiles[6][6]` is itself (`@`). One character a tile, the game's legend (`.` grass, `~` water, `T` tree, `P` a person, `c` a chicken...) |
| `beings` | people and creatures in view, nearest first, at most 8: `dx`, `dy` from it (east and south positive), `hp`, `name` (people, named creatures), `side`: `ally` (it wrote so in its memory), `foe` (a hostile kind, or one that hit it) or left out. A recording that only knows the distance has `d` instead of `dx`, `dy` |
| `inv` | what it carries, item: count |
| `goal` | the goal the mind gave the body: `kind`, `item`, `with` (a person), `since_s`, `have` (how many of the item it carries). Left out with no goal |
| `top` | the policy's six likeliest actions this decision, after the legal mask, `[name, probability]` |
| `act` | the action it chose and runs |
| `legal`, `of` | how many actions were legal, out of how many |
| `value` | the critic's estimate of what is to come, when the exported policy carries a value head (today's `export.py` does not: left out) |

### `thought`: one call of the mind (the local model), when it answers

```json
{"k": "thought", "t": 1795460440.2, "took_s": 1.5,
 "facts": "You: health 17/20, food 16/20, carrying stone_block 21 ...\ndoben00: 3 tiles away, health 14, holding sword. ...",
 "reply": {"goal": "gather", "with": "doben00", "item": "wood", "say": "wood near tree?",
           "relation_update": {"doben00": {"ally": false, "note": "staying away, no threat yet"}}}}
```

`facts` is the text the model was given, as given (`rl/mind.py`); `reply` its JSON answer, parsed (the raw text, if it
did not parse, as `{"raw": "..."}`).

### `chat`: a line heard or said

```json
{"k": "chat", "t": 1795460446.0, "from": "doben00", "text": "you good ?"}
{"k": "chat", "t": 1795460447.1, "from": "Rook", "text": "all good, digging", "said": true}
```

### `event`: something that happened near it

```json
{"k": "event", "t": 1795460525.0, "what": "hurt", "text": "sporeling hit Rook (1)", "n": 1}
```

`what` is the game's event type where there is one (`attack`, `death`, `creature_died`, `give`, `craft`, `join`,
`boss`), or `hurt` (it lost health), `goal` (the mind set a new goal). `text` is one short line for people to read.

### `memory`: its notes on people (memory.json), at the start and whenever they change

```json
{"k": "memory", "t": 1795460440.2, "people": {"doben00": {"ally": true, "notes": "gives path, takes berries"}}}
```

## Size

About 1 KB a frame (the tiles are 169 bytes), one frame a second: some 3.5 MB an hour, plain text, and it compresses
ten to one. The SSE server keeps the last 400 lines (plus the last `hello` and `memory`) for a page that connects late.
