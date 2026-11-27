"""The architect (#66): the home drawn by the model as a tile grid (rl/blueprint.py), judged (rl/judge.py, stamped in a
blank world) and drawn again with what failed, a few rounds, the best drawing kept. The judge's rules are the ones a
build is read by once it stands, so a design that passes them is laid as it is; the model only draws, the build verb
lays it. One call a round, when the mind sets a home to build, not a step of the body."""

import re

from . import blueprint

ROUNDS = 5
NAMED = 200                 # ...and for a build the mind named in its own words (a tower, a round house): its choice
BUDGET = 130                # walls and blocks, at the most: a home for one 3 by 3 inside under a gable (a flat roof: 64)

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
Or a garden: "garden": true - a plot fenced round, a gate in the middle of the "door" side, beds of tilled soil
sown with seed, "beds": "rows" (a walk between each two beds) or "full"; no furniture, no roof.
Or several of these on one site: {{"parts": [design, design, ...]}}, each part after the first with "at": the side
of those before it it goes on (a house and its garden east of it; a walled yard west of a tower), 2 to 4 parts.
A roof is made of blocks, as a builder makes one: "flat" is one layer; "gable" steps up a layer a row to a ridge
along the long side; "hip" steps up on all four sides (on a round: a dome). Its material: {materials} ("roof" is
shingles; stone or earth sits on a deck of planks); "eaves" (the lowest ring) and "ridge" (the top) in another
material if you like: a craftsman's roof mixes them.

A good home, as the game judges it: every storey closed with its door and covered; 6 to 40 tiles inside the ground
storey of a one-storey home; a bed, a campfire within 3 tiles of the bed (light, a station), a chest, a table or
chair; each piece reached on foot from its storey's door; the wall with the door an odd number of tiles long (the
door at its middle, the build mirror-symmetric). A base is a round or a yard left open (no roof): closed, a door
or gate, a campfire in it. Every wall, plank and roof tile is gathered and made by hand: at most {budget} in all
(a 3 by 3 one-storey home with a flat roof takes 64, under a gable about 130; each storey more about 40; a gable or
hip is one and a half to twice its flat roof, each step standing on the block under it; stone or earth a deck more).

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
    keys = ("width", "radius", "shape", "garden", "underground", "parts")
    return spec if isinstance(spec, dict) and any(k in spec for k in keys) else None


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


def _cost_parts(s) -> str:
    """What a design's cost is made of, so a redraw knows what to cut: " (walls 75, floors 50, roof 168)"."""
    from collections import Counter

    from nexel_agent.rl.structure import ROOF_MATERIALS
    by = Counter()
    for *_, z, p in s.cells():
        if p == "wall":
            by["walls"] += 1
        elif p in ROOF_MATERIALS:
            by["floors" if p == "plank" and z % 3 == 0 and z < 3 * getattr(s, "n", 1) else "roof"] += 1
    return " (" + ", ".join(f"{k} {v}" for k, v in by.items() if v) + ")" if by else ""


def judged(reply: str, budget: int, want: dict | None = None) -> dict:
    """The design in a reply judged, its cost a check of its own (over the budget fails it): a design's JSON as a
    build of storeys (rl/judge3d.py), a grid as drawn (blueprint.judged). `want`, what the words asked plainly
    (rl/asked.py): a design unlike it fails "words"."""
    from . import judge3d
    from .asked import unlike
    spec = spec_of(reply)
    if spec is not None:
        j = judge3d.judged(spec)
        c = j["structure"].cost() if j["ok"] else 0
        parts = _cost_parts(j["structure"]) if j["ok"] else ""
        j = {**j, "spec": spec, "told": judge3d.told(j) if j["ok"] else j["told"]}
        if j["ok"] and want:
            j["of"] += 1
            if differs := unlike(spec, want):
                j["failed"], j["told"] = j["failed"] + ["words"], (j["told"] if j["told"] != "nothing failed" else "") \
                    + ("; " if j["failed"] else "") + differs
            else:
                j["grade"] += 1
    else:
        j = blueprint.judged(grid_of(reply))
        c = cost(j["rows"]) if j["ok"] and j.get("rows") else 0
        parts = ""
    j["of"] += 1
    if j["ok"] and c > budget:
        j["failed"] = j["failed"] + ["cost"]
        j["told"] = (j["told"] if j["told"] != "nothing failed" else "") + (
            f"{'; ' if j['failed'][:-1] else ''}it takes {c} walls, planks and roof tiles{parts}, at most {budget}: "
            "cut what costs most and the words did not ask for - fewer storeys, a smaller inside, a flat roof (a gable or hip adds layers)")
    elif j["ok"]:
        j["grade"] += 1
    return j


def _rank(j) -> tuple:
    """A design read before none; within budget first (what cannot be gathered is no design to lay); then the grade."""
    return bool(j.get("ok")), "cost" not in j.get("failed", ()), j["grade"]      # (no design at all: last)


def design(brief: str, ask, rounds: int = ROUNDS, budget: int = BUDGET, words: str | None = None) -> dict:
    """The best drawing for `brief` in `rounds`: {"rows", "grade", "of", "failed", "told", "history": [grades]}.
    `ask(messages) -> reply text` is the model; `words`, what was asked for (the brief, if not given), whose plain
    asks a design keeps (rl/asked.py)."""
    from .asked import asked
    from .structure import ROOF_MATERIALS, UPSTAIRS_FURNITURE
    want = asked(brief if words is None else words)
    msgs = [{"role": "user", "content": DRAW.format(brief=brief, furniture=", ".join(blueprint.FURNITURE), budget=budget,
                                                    upstairs=", ".join(UPSTAIRS_FURNITURE),
                                                    materials=", ".join(ROOF_MATERIALS))}]
    best, history = None, []
    for _ in range(rounds):
        reply = ask(msgs)
        j = judged(reply, budget, want)
        history.append(j["grade"])
        if best is None or _rank(j) > _rank(best):
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


STUFF = ("wood", "stone", "fiber", "plank", "roof", "stone_block", "dirt", "wood_wall", "stone_wall", "door", "fence",
         "gate", "bed", "campfire", "chest", "table", "chair", "workbench", "seed", "hoe", "pickaxe", "stairs")


def carried(obs) -> str:
    """What the builder carries of what builds are made of, for the brief: a design of it goes up sooner (live trial
    3: a two-storey house drawn with 69 wood still to gather, and 25 minutes went by gathering it)."""
    inv = (obs.get("self") or {}).get("inventory") or {}
    have = [f"{inv[k]} {k}" for k in STUFF if inv.get(k)]
    return (". What they carry now: " + ", ".join(have) + " (what it is built of is gathered first: a design of what they"
            " carry goes up sooner)") if have else ". They carry nothing to build with yet: a small design goes up sooner"


def plan_from(kept: dict):
    """The plan of a design kept: a build of storeys (its JSON), or a grid."""
    from .build import drawn
    from .plan import of_structure
    from .compound import describe
    from .dug import design_of
    if not kept.get("spec"):
        return drawn(kept["rows"])
    plan = of_structure(design_of(kept["spec"]))
    plan.words = describe(kept["spec"])                     # (what it is, told in the mind's facts: rl/home.py)
    return plan


class Drafting:
    """The live side: when the mind builds, the architect draws first, in a thread, and the body waits for the drawing
    (it would site the hut otherwise). For a home not begun: the mind's words for it, or a home for one. Once what was
    drawn stands, the mind naming another build in its words ("a vegetable garden by the house") has it drawn as a
    part set beside it (rl/compound.py): the same site, what stands laid already. A drawing at full marks is the plan
    (social.plan), kept in `path` (plan.json, by memory.json: laid as drawn the next session too); short of full marks,
    or no model, nothing changes (the hut, or what stands). Words that drew nothing are not drawn again. The mind's
    words for a build are drawn over a hut the body began on its own."""

    def __init__(self, ask, path, who: str = "", brief: str = "a home for one"):
        import json
        self.ask, self.path, self.who, self.brief, self.task, self.want = ask, path, who, brief, None, None
        self.kept = json.loads(path.read_text()) if path and path.exists() else None      # ({"spec"} or {"rows"})
        self.tried = set()                     # (words drawn already, whatever came of them; "": the first home)
        self.last = None                       # the last design: {"grade", "of", "history"} (logged by live.py)
        self.brief_for = brief

    def holds(self, social, building: bool, obs, stats=None, words=None) -> bool:
        """True while the body is to wait for a drawing; the plan set on social once it is decided (and the design
        told in `stats`, the run's, as "draft")."""
        import concurrent.futures
        if self.kept and getattr(social, "plan", None) is None:
            social.plan = plan_from(self.kept)
        if self.task is not None:              # (a drawing comes back whether the mind still builds or not: kept
            if not self.task.done():           # for when it builds again)
                return building
            self._finish(social, stats)
            return False
        if not building:
            return False
        want = self._wanted(social, obs, words)
        if want is None:
            return False
        self.want, self.brief = want, want[1] + (f". It is for: {self.who}" if self.who else "") + carried(obs)
        self.tried.add(words or "")
        self.task = concurrent.futures.ThreadPoolExecutor(1).submit(design, self.brief, self.ask, ROUNDS, want[2], want[1])
        return True

    def _wanted(self, social, obs, words):
        """("first" | "part", the brief, the budget), or None: nothing to draw now."""
        from .build import laid_by, plan_of
        if (words or "") in self.tried:
            return None
        plan = plan_of(social)
        if not self.kept:                      # (a build the mind names in its words is drawn over a hut the body
            if words:                          # began on its own - live trial 1: the body sited a hut before the
                return ("first", words, NAMED) # mind asked for "a round house with a domed roof", and that was lost)
            begun = social.home and any(laid_by(obs, social, plan)(c) for c in plan.cells)
            return None if begun else ("first", self.brief_for, BUDGET)
        if words and self.kept.get("spec") and social.home and all(laid_by(obs, social, plan)(c) for c in plan.cells):
            return ("part", f"{words}; it goes beside a build that stands already: one design, its \"at\" the side it "
                            "goes on", NAMED)
        return None

    def _finish(self, social, stats):
        import json

        from .dug import design_of
        kind, done, self.task = self.want[0], self.task, None
        try:
            d = done.result()
        except Exception as e:                 # (the model gone or broken: nothing changes)
            d = {"error": repr(e)[:200]}
        self.last = {k: d.get(k) for k in ("grade", "of", "history", "rows", "spec", "error") if k in d}
        if stats is not None:
            stats["draft"] = self.last
        if not d.get("ok") or d["grade"] != d["of"]:
            return
        if kind == "first":
            self.kept = {"spec": d["spec"]} if d.get("spec") else {"rows": d["rows"]}
            social.home = None                                                       # (sited anew, for its shape)
        else:
            spec = self.kept["spec"]
            parts = spec["parts"] if spec.get("parts") else [spec]
            new = {"parts": [*parts, {"at": "east", **d["spec"]}]}
            try:
                design_of(new)
            except (ValueError, KeyError, TypeError):
                return
            self.kept = {"spec": new}
        social.plan = plan_from(self.kept)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({**self.kept, "brief": self.brief, "grade": d["grade"]}))
