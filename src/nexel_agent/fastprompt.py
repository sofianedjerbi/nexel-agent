"""The real-time thinker's words: a short stable prefix (cached by the model server), a compact observation,
and the fast chat reply's tiny prompt. The plan format and its parser: plans.py.
"""

import json
import re

from .ladder import NEEDS_TOOL as NEEDS, ladder_text, progress_lines
from .policy import DIRS, reachable
from .plans import _kind, _optional
from .senses import rules_text
from .sim.specs import LEGEND, TOOL_SPECS
from .tiles import RECIPES

CORE_RECIPES = ("workbench", "plank", "campfire", "torch", "rope", "axe", "pickaxe", "hoe", "spear", "chest", "bed",
                "fence", "door", "wood_wall", "stone_wall", "roof", "path", "cooked_meat", "bread", "ingot", "sword", "boat")
PREFIX_RECIPES = ("workbench", "campfire", "axe", "pickaxe", "hoe", "wood_wall", "door", "roof", "chest", "bed",
                  "ingot", "sword", "cooked_meat")   # (the rest: "Can craft now" and "next" in the view)
STATIONS = {"workbench": "&", "campfire": "^"}
GROUNDS_SKIP = set(".,o_'nu-:")            # plain ground: not worth a line in "Near"
HARVEST = set("TbM*HQFvUtw")                # what digging gathers (trees, bushes, rocks, ore, mushrooms, ripe crops)
VIEW = 5                                   # the map crop's radius (11x11)


def signature(tool: str) -> str:
    parts = [tool]
    for k, v in TOOL_SPECS[tool]["args"].items():
        word = "D" if _kind(k, v) == "dir" else k.upper() if k not in ("with", "run", "list") else k
        parts.append(f"[{word}]" if _optional(v) or k in ("with", "list") else word)
    return " ".join(parts)


TOOL_HINTS = {
    "move_to": "walk there (finds the way)", "dig": "harvest/dig the thing next to you that way (with: axe, pickaxe, hoe)",
    "place": "put down a thing you carry next to you, e.g. place S workbench (things, not wood or stone)", "craft": "make a recipe", "attack": "hit within 3 tiles",
    "say": "talk to people near", "jump": "onto a step 1 higher", "look_far": "see 48 tiles that way",
}


RARE = {"milk", "feed", "name", "post", "open_mail", "unroof", "read_sign", "water", "stop", "dig_down", "leap", "unwear",
        "write_sign", "shout", "ignite"}   # (not listed: tokens)


def stable_prefix(name: str, persona: str) -> str:
    """The system message: the same bytes every call (the server caches it), so it is paid for once."""
    tools = "\n".join(f"- {signature(t)}: {TOOL_HINTS[t]}" for t in TOOL_SPECS if t in TOOL_HINTS)
    tools += "\n- also: " + "; ".join(signature(t) for t in TOOL_SPECS if t not in TOOL_HINTS and t not in RARE)
    tools += "; remember TEXT; forget N; drop_card N; skill NAME [ARGS] (one of your skills, listed when you have some)"
    tools += ("\n- verbs (one intent, the actions it takes): dig_at TARGET [with T]; fight TARGET [with W]; go_next_to TARGET; move_away TARGET [N];"
              " place_near ITEM; craft_here RECIPE [N]; where TARGET. TARGET: x y, #id, a name or a kind (tree, rock, Ana)")
    recipes = " ".join(f"{'By hand' if not st else '@' + st}: " + "; ".join(
        f"{n}=" + "+".join(f"{c} {k}" for k, c in RECIPES[n].inputs.items()) for n in PREFIX_RECIPES
        if n in RECIPES and RECIPES[n].station == st) + "." for st in (None, "workbench", "campfire"))
    return f"""You are {name}, living in a wild island world with no laws. Who you are: {persona}
You play in real time: answer with a short plan; it runs at once while you think of the next one.
World: top-down grid, north is up (y-1), coordinates are absolute (x,y). Health 0-20, food 0-20 (work uses food).
To gather, stand next to a thing and dig toward it: tree -> wood, berry_bush -> berries + seed (it becomes a bare_bush), bare_bush -> fiber,
rock -> stone (slow by hand), ore -> ore (pickaxe). Items next to you come to you. Wood and stone are materials,
not things to place. @workbench crafts need a placed workbench within 2 tiles of you. Seeds go in tilled soil (dig ground with a hoe). Water: you swim. A step one
higher: jump D. {rules_text()}
Commands (D is N, E, S or W; [x] optional):
{tools}
Recipes: {recipes}
Answer with ONE JSON object and nothing else:
{{"goal":"<2-5 words>","done_when":{{<test>}},"do":["<command>", 2 to 6 commands]}}
done_when (your goal's test, checked after each action, all keys must hold): "have": {{"wood": 8}} | "placed": "workbench" |
"near": "workbench" | "at": [x, y] | "hp_at_least": 15. A plan runs to its end, its first failure or a card;
"repeat": N plays it again up to N times while done_when does not hold.
Example: {{"goal":"get wood","done_when":{{"have":{{"wood":6}}}},"do":["dig_at tree","dig_at tree"]}} (a dig takes a whole tree)
"remember TEXT" keeps a note of yours (a place, a person, a promise); "forget N" drops note [N]."""


def craftable(inv: dict, grid: list[str]) -> list[str]:
    """Recipes the inventory pays for now, a station recipe marked when its station is not within 2 tiles."""
    r = len(grid) // 2
    near = {grid[r + dy][r + dx] for dy in range(-2, 3) for dx in range(-2, 3)
            if 0 <= r + dy < len(grid) and 0 <= r + dx < len(grid)}
    out = []
    for name in sorted(RECIPES, key=lambda n: CORE_RECIPES.index(n) if n in CORE_RECIPES else 99):
        rec = RECIPES[name]
        if all(inv.get(k, 0) >= n for k, n in rec.inputs.items()):
            st = rec.station
            out.append(name if not st or STATIONS.get(st) in near else f"{name}(@{st})")
    return out[:5]


def _via(obs, reach, tx, ty):
    """Where to stand to work on (tx, ty), and the dir to face: nearest walkable side."""
    me = obs["self"]
    x, y = me["x"], me["y"]
    best = None
    for d, (ddx, ddy) in DIRS.items():
        sx, sy = tx - ddx, ty - ddy                      # stand there, facing d, the target is in front
        q = (sx - x, sy - y)
        if q in reach and (best is None or reach[q] < best[0]):
            best = (reach[q], sx, sy, d)
    return best


def near_lines(obs, limit=5) -> list[str]:
    """The nearest thing of each kind in the full view, with where to stand and which way to face."""
    me, grid = obs["self"], obs["grid"]
    r = len(grid) // 2
    x0, y0 = me["x"] - r, me["y"] - r
    reach = reachable(obs)
    best: dict[str, tuple] = {}
    for j, row in enumerate(grid):
        for i, ch in enumerate(row):
            if ch in GROUNDS_SKIP or ch in "@ " or ch in "PcksDIGSiE$":
                continue
            d = abs(i - r) + abs(j - r)
            if ch not in best or d < best[ch][0]:
                best[ch] = (d, x0 + i, y0 + j)
    out = []
    for ch, (d, tx, ty) in sorted(best.items(), key=lambda kv: kv[1][0])[:limit]:
        name = LEGEND.get(ch, ch).split(" (")[0]
        v = _via(obs, reach, tx, ty)
        need = NEEDS.get(ch)
        if need and not me["inventory"].get(need):
            out.append(f"{name} ({tx},{ty}): needs a {need}")
        elif ch not in HARVEST:
            out.append(f"{name} ({tx},{ty})" + ("" if not v else " next to you, " + v[3] if v[0] == 0
                                                   else f": stand at ({v[1]},{v[2]}), it is {v[3]}"))
        elif v and v[0] == 0:
            out.append(f"{name} ({tx},{ty}) next to you, {v[3]}")
        elif v:
            out.append(f"{name} ({tx},{ty}): stand at ({v[1]},{v[2]}), it is {v[3]}")
        else:
            out.append(f"{name} ({tx},{ty}) {d} away, no way seen")
    return out


def needs_for(goal: str | None, inv: dict, grid) -> str:
    """What the goal's recipe still lacks (the first recipe named in the goal): 'pickaxe needs: 3 stone (have 0), ...'."""
    words = (goal or "").lower().replace("-", " ").split()
    name = next((f for w in words for f in (w, w[:-1], w[:-2]) if f in RECIPES), None)   # (torches -> torch)
    if not name:
        return ""
    rec = RECIPES[name]
    parts = [f"{n} {k} (have {inv.get(k, 0)})" for k, n in rec.inputs.items()]
    if rec.station:
        r = len(grid) // 2
        near = any(grid[r + dy][r + dx] == STATIONS.get(rec.station) for dy in range(-2, 3) for dx in range(-2, 3))
        parts.append(f"a {rec.station} within 2 tiles ({'yes' if near else 'none'})")
    return f"{name} needs: " + ", ".join(parts) + "."


def compact_obs(obs, goal=None, heard=(), history=(), pending=(), notes="", lessons="", habit="", record=None,
                danger=(), recall=(), goal_line="", skills="") -> str:
    me, grid = obs["self"], obs["grid"]
    r = len(grid) // 2
    crop = [row[r - VIEW:r + VIEW + 1] for row in grid[r - VIEW:r + VIEW + 1]]
    inv = me["inventory"]
    used = sorted({c for row in crop for c in row} - {"@"})
    flags = [f for f, on in (("sheltered", me.get("sheltered")), ("in a cave", me.get("in_cave")),
                             ("swimming", me.get("swimming"))) if on]
    lines = [f"Time {obs['time']}. You: ({me['x']},{me['y']}) z{me.get('z', 0)} on {me['standing_on']}"
             + (", " + ", ".join(flags) if flags else "") + f". Health {me['hp']}/20, food {me['hunger']}/20.",
             *danger,
             "Carry: " + (", ".join(f"{k} {v}" for k, v in inv.items()) or "nothing") + ".",
             *([f"Can craft now: {', '.join(c)}."] if (c := craftable(inv, grid)) else []),
             *([goal_line] if goal_line else []),
             *progress_lines(obs, goal, record),
             *([f"Your skills (\"skill NAME [N]\" does the whole of it, walking and gathering included): {skills}."]
               if skills else []),
             f"Map (@ = you; rows y{me['y'] - VIEW}..{me['y'] + VIEW}, columns x{me['x'] - VIEW}..{me['x'] + VIEW}):",
             *crop,
             "Key: " + ", ".join(f"{c} {LEGEND.get(c, c).split(' (')[0]}" for c in used if c in LEGEND)]
    near = near_lines(obs)
    if near:
        lines.append("Near: " + "; ".join(near))
    if obs["items"]:
        lines.append("On the ground: " + "; ".join(
            f"({me['x'] + i['dx']},{me['y'] + i['dy']}) " + ", ".join(f"{k} {v}" for k, v in i["items"].items())
            for i in obs["items"][:3]))
    if obs.get("signs"):
        lines.append("Signs: " + "; ".join(f"({me['x'] + s['dx']},{me['y'] + s['dy']}) \"{s['text'][:60]}\""
                                           for s in obs["signs"][:2]))
    if heard:
        lines.append("Heard: " + " | ".join(heard))
    if history:
        lines.append("Done: " + " | ".join(history))
    if pending:
        lines.append("Already queued (plan what comes AFTER): " + ", ".join(pending))
    if recall:
        lines.append("You remember: " + " | ".join(recall))
    if notes:
        lines.append("Your notes: " + notes)
    if lessons:
        lines.append("Learned: " + lessons)
    if habit:
        lines.append(habit)
    lines.append("Your plan? /no_think")
    return "\n".join(lines)


def chat_messages(name: str, persona: str, goal: str, speaker: str, text: str, recent=(), me=None) -> list[dict]:
    """A real player talks to you: answer what they said (a question gets its answer), in your voice but plainly, never a
    riddle ("gatsu is full roleplaying, not answering my messages": "Hmph. Coming." and it never came). Then, on a line
    of its own, what you do about it: DO: come | stop | nothing."""
    me = me or {}
    inv = ", ".join(f"{k} {v}" for k, v in sorted((me.get("inventory") or {}).items(), key=lambda kv: -kv[1])[:6]) or "nothing"
    sys_ = (f"You are {name}, a player in a survival game world. Your character: {persona}\nA real person playing the game "
            "talks to you. Talk like a normal, friendly person in a chat: short, natural, a hint of your character at "
            "most. Answer what they said; say what you do or have in plain words (no coordinates, no counts unless "
            "asked). Never menacing, never cryptic, never weird, never repeat their words. At most 20 words.\n"
            'Reply with JSON only: {"say": "<your line>", "do": "come" | "stop" | "nothing"}. come: only if they ask you '
            "to come, follow or help them. stop: they ask you to stop or to do something else. nothing: greetings, "
            "questions, anything else.\nExamples:\n"
            'hi -> {"say": "Hey! Just chopping some wood. You?", "do": "nothing"}\n'
            'can you help me dig? -> {"say": "Sure, on my way.", "do": "come"}\n'
            'stop cutting trees -> {"say": "Alright, I\'ll find something else to do.", "do": "stop"}')
    ctx = ("Earlier: " + " | ".join(recent) + "\n") if recent else ""
    status = f"You are at ({me.get('x', '?')}, {me.get('y', '?')}), health {me.get('hp', '?')}, food {me.get('hunger', '?')}/20, carrying: {inv}."
    return [{"role": "system", "content": sys_},
            {"role": "user", "content": f"{ctx}You are busy with: {goal or 'getting by'}. {status}\n{speaker} says: {text}\n/no_think"}]


INTENTS = ("follow", "stop", "give", "trade", "fight", "flee")


def chat_parse(out: str) -> tuple[str, str]:
    """(what to say, what to do: come | stop | ""): the reply's JSON; failing that its text, a DO: in it read apart."""
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        d = None
    if isinstance(d, dict):
        say, do = str(d.get("say") or ""), str(d.get("intent") or d.get("do") or "").lower()
    else:
        k = re.search(r"\**\s*DO\s*:\s*(\w+)", out, re.I)
        say, do = (out[:k.start()] if k else out), (k.group(1).lower() if k else "")
    say = re.sub(r"^\s*[\w.-]+\d\s*:\s*", "", " ".join(say.split())).strip().strip('"').strip()   # (not "gatsu0: ...")
    do = {"come": "follow"}.get(do, do)
    return say[:160], do if do in INTENTS else ""
