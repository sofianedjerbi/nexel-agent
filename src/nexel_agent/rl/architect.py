"""The architect (#66): the home drawn by the model as a tile grid (rl/blueprint.py), judged (rl/judge.py, stamped in a
blank world) and drawn again with what failed, a few rounds, the best drawing kept. The judge's rules are the ones a
build is read by once it stands, so a design that passes them is laid as it is; the model only draws, the build verb
lays it. One call a round, when the mind sets a home to build, not a step of the body."""

import re

from . import blueprint

ROUNDS = 4

DRAW = """You design small homes for a top-down tile game. Draw one as a grid, north up, one character a tile:

  #  wall     D  door      .  floor (inside)      =  path (outside, cobble)
  b  bed      f  campfire  c  chest   t  table   h  chair   k  workbench
  w  window   (a space)  outside, left as it is

The roof is added over the walls and one tile past them; do not draw it. At most {most} by {most} tiles.
A good home, as the game judges it: the walls close a room (a door is part of the wall; corners count), a door in
the middle of a wall, opening onto open ground, a path of two or more '=' leading out from the door, 6 to 40 floor
tiles inside, a bed, a campfire near the bed (light and a station), a chest, a table or chair, each beside floor
one can walk to from the door, one clean rectangle, straight walls, and mirror-symmetric about the door's line.
A window is a gap to the game (the room no longer counts as closed): leave windows out. Keep the tile inside the
door free, so one can walk in.

Brief: {brief}

Answer with the grid only, in a ``` block."""

AGAIN = """Judged {grade} of {of}. What failed: {told}.
Draw it again, fixing that, keeping what passed. The grid only, in a ``` block."""


def grid_of(reply: str) -> str:
    """The grid in a reply: the ``` block if there is one, else the reply."""
    m = re.search(r"```[a-z]*\n(.*?)```", reply or "", re.S)
    return m.group(1) if m else (reply or "")


def design(brief: str, ask, rounds: int = ROUNDS) -> dict:
    """The best drawing for `brief` in `rounds`: {"rows", "grade", "of", "failed", "told", "history": [grades]}.
    `ask(messages) -> reply text` is the model."""
    msgs = [{"role": "user", "content": DRAW.format(brief=brief, most=blueprint.MOST)}]
    best, history = None, []
    for _ in range(rounds):
        reply = ask(msgs)
        j = blueprint.judged(grid_of(reply))
        history.append(j["grade"])
        if best is None or j["grade"] > best["grade"]:
            best = j
        if j["ok"] and not j["failed"]:
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": AGAIN.format(grade=j["grade"], of=j["of"], told=j["told"])}]
    return {**best, "history": history}


def model(url: str, *, max_tokens: int = 600, temperature: float = 0.4):
    """`ask` on an OpenAI-style chat server (the local model server, rl/live.py MODEL_URL)."""
    import httpx

    def ask(msgs):
        r = httpx.post(url, timeout=300, json={"model": "default_model", "messages": msgs, "max_tokens": max_tokens,
                                               "temperature": temperature})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return ask
