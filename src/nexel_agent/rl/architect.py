"""The architect (#66): the home drawn by the model as a tile grid (rl/blueprint.py), judged (rl/judge.py, stamped in a
blank world) and drawn again with what failed, a few rounds, the best drawing kept. The judge's rules are the ones a
build is read by once it stands, so a design that passes them is laid as it is; the model only draws, the build verb
lays it. One call a round, when the mind sets a home to build, not a step of the body."""

import re

from . import blueprint

ROUNDS = 4
NAMED = 200                 # ...and for a build the mind named in its own words (a tower, a round house): its choice
BUDGET = 82                 # walls and roof tiles, at the most: a home for one 3 by 5 inside (the hut, 3 by 3: 64)

DRAW = """You design builds for a tile game seen from above, built in storeys. Choose the design; it is drawn
for you: walls all round the inside, a door in the middle of the wall you name, a path out of it, a floor of
planks over each storey that has one above it, a flight of stairs outside up to each upper storey's own door, a
roof over the top (one tile past its walls) unless you leave it open.

Answer with JSON only:
{{"shape": "rect" or "round", "width": tiles west to east, "depth": tiles north to south (rect),
  "radius": 2 to 6 (round), "storeys": 1 to 3, "door": "north"|"south"|"east"|"west",
  "roof": false (open) or {{"style": "flat"|"gable"|"hip", "material": ..., "eaves": ..., "ridge": ...}},
  "furniture": {{"bed": [x, y], "table": [x, y, storey], ...}}, "path": 2 or 3}}
Places are from the inside's top-left (x west to east, y north to south; a round's inside is its box), storey 0
the ground (the default); two of a piece: "bed": [[x, y], [x, y]]. Furniture: {furniture}; on an upper storey
only {upstairs} (the game stands nothing else on a floor).
Or dug into the land: "underground": true - a trench stepping down to its door and a room tunnelled behind it,
the land its walls and its roof (no storeys, no roof to choose; nothing to gather: the land dug is the walls).
A roof is made of blocks, as a builder makes one: "flat" is one layer; "gable" steps up a layer a row to a ridge
along the long side; "hip" steps up on all four sides (on a round: a dome). Its material: {materials} ("roof" is
shingles; stone or earth sits on a deck of planks); "eaves" (the lowest ring) and "ridge" (the top) in another
material if you like: a craftsman's roof mixes them.

A good home, as the game judges it: every storey closed with its door and covered; 6 to 40 tiles inside the ground
storey of a one-storey home; a bed, a campfire within 3 tiles of the bed (light, a station), a chest, a table or
chair; each piece reached on foot from its storey's door; the wall with the door an odd number of tiles long (the
door at its middle, the build mirror-symmetric). A base is a round or a yard left open (no roof): closed, a door
or gate, a campfire in it. Every wall, plank and roof tile is gathered and made by hand: at most {budget} in all
(a 3 by 3 one-storey home with a flat roof takes 64; each storey more about 40; a gable or hip about half again
its flat roof; stone or earth a deck more).

Brief: {brief}"""

AGAIN = """Judged {grade} of {of}. What failed: {told}.
Choose again, fixing that, keeping what passed. JSON only."""


def spec_of(reply: str):
    """The design's JSON in a reply, or None (a grid, or nothing)."""
    import json
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        spec = json.loads(m.group(0))
    except ValueError:
        return None
    return spec if isinstance(spec, dict) and ("width" in spec or "radius" in spec) else None


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
    """The design in a reply judged, its cost a check of its own (over the budget fails it): a design's JSON as a
    build of storeys (rl/judge3d.py), a grid as drawn (blueprint.judged)."""
    from . import judge3d
    spec = spec_of(reply)
    if spec is not None:
        j = judge3d.judged(spec)
        c = j["structure"].cost() if j["ok"] else 0
        j = {**j, "spec": spec, "told": judge3d.told(j) if j["ok"] else j["told"]}
    else:
        j = blueprint.judged(grid_of(reply))
        c = cost(j["rows"]) if j["ok"] and j.get("rows") else 0
    j["of"] += 1
    if j["ok"] and c > budget:
        j["failed"] = j["failed"] + ["cost"]
        j["told"] = (j["told"] if j["told"] != "nothing failed" else "") + \
            f"{'; ' if j['failed'][:-1] else ''}it takes {c} walls, planks and roof tiles, at most {budget}: smaller"
    elif j["ok"]:
        j["grade"] += 1
    return j


def design(brief: str, ask, rounds: int = ROUNDS, budget: int = BUDGET) -> dict:
    """The best drawing for `brief` in `rounds`: {"rows", "grade", "of", "failed", "told", "history": [grades]}.
    `ask(messages) -> reply text` is the model."""
    from .structure import ROOF_MATERIALS, UPSTAIRS_FURNITURE
    msgs = [{"role": "user", "content": DRAW.format(brief=brief, furniture=", ".join(blueprint.FURNITURE), budget=budget,
                                                    upstairs=", ".join(UPSTAIRS_FURNITURE),
                                                    materials=", ".join(ROOF_MATERIALS))}]
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


def plan_from(kept: dict):
    """The plan of a design kept: a build of storeys (its JSON), or a grid."""
    from .build import drawn
    from .plan import of_structure
    from .dug import design_of
    return of_structure(design_of(kept["spec"])) if kept.get("spec") else drawn(kept["rows"])


class Drafting:
    """The live side: when the mind builds a home not begun, the architect draws it first, in a thread, and the body
    waits for the drawing (it would site the hut otherwise). A drawing at full marks is the plan (social.plan) and is
    kept in `path` (plan.json, by memory.json: a home drawn is laid as drawn the next session too); short of full
    marks, or no model, the hut. A home begun is never drawn again."""

    def __init__(self, ask, path, who: str = "", brief: str = "a home for one"):
        import json
        self.ask, self.path, self.who, self.brief, self.task, self.done = ask, path, who, brief, None, False
        self.kept = json.loads(path.read_text()) if path and path.exists() else None      # ({"spec"} or {"rows"})
        self.last = None                       # the last design: {"grade", "of", "history"} (logged by live.py)

    def holds(self, social, building: bool, obs, stats=None, words=None) -> bool:
        """True while the body is to wait for the drawing; the plan set on social once it is decided (and the design
        told in `stats`, the run's, as "draft")."""
        import concurrent.futures
        import json

        from .build import laid, plan_of
        if self.kept and getattr(social, "plan", None) is None:
            social.plan = plan_from(self.kept)
        if not building or self.done or getattr(social, "plan", None) is not None:
            return False
        if social.home and any(laid(obs, social.home[0] + dx, social.home[1] + dy, p) for dx, dy, p in plan_of(social).pieces):
            self.done = True                   # (the hut begun: laid as it is)
            return False
        if self.task is None:                  # (the mind's own words for the build, when it said them: the brief)
            self.brief = (words or self.brief) + (f". It is for: {self.who}" if self.who else "")
            self.task = concurrent.futures.ThreadPoolExecutor(1).submit(design, self.brief, self.ask, ROUNDS,
                                                                        NAMED if words else BUDGET)
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
        self.last = {k: d.get(k) for k in ("grade", "of", "history", "rows", "spec")}
        if stats is not None:
            stats["draft"] = self.last
        if d.get("ok") and d["grade"] == d["of"]:
            self.kept = {"spec": d["spec"]} if d.get("spec") else {"rows": d["rows"]}
            social.plan, social.home = plan_from(self.kept), None                      # (sited anew, for its shape)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps({**self.kept, "brief": self.brief, "grade": d["grade"]}))
        return False
