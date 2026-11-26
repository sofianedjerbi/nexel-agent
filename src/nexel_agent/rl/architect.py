"""The architect (#66): the home drawn by the model as a tile grid (rl/blueprint.py), judged (rl/judge.py, stamped in a
blank world) and drawn again with what failed, a few rounds, the best drawing kept. The judge's rules are the ones a
build is read by once it stands, so a design that passes them is laid as it is; the model only draws, the build verb
lays it. One call a round, when the mind sets a home to build, not a step of the body."""

import re

from . import blueprint

ROUNDS = 4
BUDGET = 82                 # walls and roof tiles, at the most: a home for one 3 by 5 inside (the hut, 3 by 3: 64)

DRAW = """You design small homes for a top-down tile game. Choose the design; it is drawn for you: walls all
round the inside, the door in the middle of the wall you name, a path out of it, a roof over it all.

Answer with JSON only:
{{"width": inside tiles west to east, "depth": inside tiles north to south, "door": "north"|"south"|"east"|"west",
  "furniture": {{"bed": [x, y], ...}}, "path": 2 or 3}}
x from 0 (west) to width-1, y from 0 (north) to depth-1; two of a piece: "bed": [[x, y], [x, y]].
Furniture: {furniture}.

A good home, as the game judges it: 6 to 40 tiles inside; a bed, a campfire within 3 tiles of the bed (light, and a
station to make things at), a chest, a table or chair; every piece beside a free tile one walks to from the door;
the tile just inside the door left free; the wall with the door an odd number of tiles long (so the door is at its
middle and the home is mirror-symmetric). Every wall and roof tile is gathered and made by hand: at most {budget}
walls and roof tiles in all (the roof covers the walls and one tile past them: a 3 by 3 inside takes 64, 3 by 5
takes 82, 5 by 5 takes 104).

Brief: {brief}"""

AGAIN = """Judged {grade} of {of}. What failed: {told}.
Choose again, fixing that, keeping what passed. JSON only."""


def grid_of(reply: str) -> str:
    """The grid in a reply: a design's JSON drawn (blueprint.render), else the ``` block if there is one, else the
    reply. A design that cannot be drawn is told as what is wrong with it."""
    import json
    text = reply or ""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            spec = json.loads(m.group(0))
        except ValueError:
            spec = None
        if isinstance(spec, dict) and "width" in spec:
            try:
                blueprint.render(spec)                     # (what cannot be drawn is told as it was chosen)
                return blueprint.render(blueprint.tidied(spec))
            except ValueError as e:
                return f"? {e}"
    m = re.search(r"```[a-z]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def cost(rows) -> int:
    """The walls and roof tiles a drawing takes (what is gathered and made most)."""
    return sum(p in ("wall", "roof") for _, _, p in blueprint.pieces(rows))


def judged(reply: str, budget: int) -> dict:
    """The drawing in a reply judged (blueprint.judged), its cost a check of its own: over the budget fails it."""
    j = blueprint.judged(grid_of(reply))
    j["of"] += 1
    if j["ok"] and j.get("rows") and (c := cost(j["rows"])) > budget:
        j["failed"] = j["failed"] + ["cost"]
        j["told"] = (j["told"] if j["told"] != "nothing failed" else "") + \
            f"{'; ' if j['failed'][:-1] else ''}it takes {c} walls and roof tiles, at most {budget}: smaller"
    elif j["ok"]:
        j["grade"] += 1
    return j


def design(brief: str, ask, rounds: int = ROUNDS, budget: int = BUDGET) -> dict:
    """The best drawing for `brief` in `rounds`: {"rows", "grade", "of", "failed", "told", "history": [grades]}.
    `ask(messages) -> reply text` is the model."""
    msgs = [{"role": "user", "content": DRAW.format(brief=brief, furniture=", ".join(blueprint.FURNITURE), budget=budget)}]
    best, history = None, []
    for _ in range(rounds):
        reply = ask(msgs)
        j = judged(reply, budget)
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


class Drafting:
    """The live side: when the mind builds a home not begun, the architect draws it first, in a thread, and the body
    waits for the drawing (it would site the hut otherwise). A drawing at full marks is the plan (social.plan) and is
    kept in `path` (plan.json, by memory.json: a home drawn is laid as drawn the next session too); short of full
    marks, or no model, the hut. A home begun is never drawn again."""

    def __init__(self, ask, path, brief: str):
        import json
        self.ask, self.path, self.brief, self.task, self.done = ask, path, brief, None, False
        self.rows = json.loads(path.read_text()).get("rows") if path and path.exists() else None
        self.last = None                       # the last design: {"grade", "of", "history"} (logged by live.py)

    def holds(self, social, building: bool, obs, stats=None) -> bool:
        """True while the body is to wait for the drawing; the plan set on social once it is decided (and the design
        told in `stats`, the run's, as "draft")."""
        import concurrent.futures
        import json

        from .build import drawn, laid, plan_of
        if self.rows and getattr(social, "plan", None) is None:
            social.plan = drawn(self.rows)
        if not building or self.done or getattr(social, "plan", None) is not None:
            return False
        if social.home and any(laid(obs, social.home[0] + dx, social.home[1] + dy, p) for dx, dy, p in plan_of(social).pieces):
            self.done = True                   # (the hut begun: laid as it is)
            return False
        if self.task is None:
            self.task = concurrent.futures.ThreadPoolExecutor(1).submit(design, self.brief, self.ask)
            return True
        if not self.task.done():
            return True
        self.done = True
        try:
            d = self.task.result()
        except Exception as e:                 # (the model gone or broken: the hut)
            self.last = {"error": repr(e)[:200]}
            if stats is not None:
                stats["draft"] = self.last
            return False
        self.last = {k: d.get(k) for k in ("grade", "of", "history", "rows")}
        if stats is not None:
            stats["draft"] = self.last
        if d.get("ok") and d["grade"] == d["of"]:
            self.rows, social.plan, social.home = d["rows"], drawn(d["rows"]), None     # (sited anew, for its shape)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps({"rows": self.rows, "brief": self.brief, "grade": d["grade"]}))
        return False
