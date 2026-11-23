"""Verbs: the HOW of one intent, never the when or the what (design 3.1). Each takes its target as an argument and
plays plain game actions for it, like move_to's pathfinding does:

    go_next_to TARGET      walk to a free side of it; says which way it is      (TARGET: "x y", a name, a kind or #id)
    dig_at TARGET [with T] go next to it (however far: in view, or where it was last seen) and dig it
    move_away TARGET [N]   walk to a reachable spot at least N tiles (default 8) from it
    fight TARGET [with W]  go next to a being and attack it
    place_near ITEM        put ITEM on a free flat side of you, or of the nearest reachable tile that has one
    craft_here RECIPE [N]  craft; with its station placed in view or seen before but not within 2 tiles, walk to it first
    where TARGET / find NAME   where something is (no action)

Whether to do any of it is the model's choice.
"""

import math

from .ladder import free_dirs
from .policy import reachable
from .sim.specs import LEGEND
from .tiles import DIRS, RECIPES, TILL_TOOL

VERB_SPECS = {v: {"args": {"text": "str"}} for v in ("go_next_to", "dig_at", "move_away", "place_near", "craft_here",
                                                     "where", "find", "fight")}
TOOLS = {"axe", "pickaxe", "hoe", "spear", "sword", "watering_can", TILL_TOOL}


def locate(obs, target: str) -> list[tuple]:
    """Absolute (x, y) of what `target` names, nearest first: "x y", "#id", a being's name or kind, a tile's name."""
    me = obs["self"]
    w = target.replace(",", " ").split()
    if len(w) >= 2 and all(p.lstrip("-").isdigit() for p in w[:2]):
        return [(int(w[0]), int(w[1]))]
    name = " ".join(w).lstrip("#").lower()
    out = []
    for e in obs["entities"]:
        if name in (str(e["id"]), str(e.get("name", "")).lower(), e["kind"]):
            out.append((me["x"] + e["dx"], me["y"] + e["dy"]))
    if not out:
        g, r = obs["grid"], len(obs["grid"]) // 2
        for j, row in enumerate(g):
            for i, ch in enumerate(row):
                if ch != "@" and LEGEND.get(ch, "").split(" (")[0].lower().replace(" ", "_") in (name, name.replace(" ", "_")):
                    out.append((me["x"] - r + i, me["y"] - r + j))
    return sorted(set(out), key=lambda p: math.dist(p, (me["x"], me["y"])))


def note_seen(obs, seen: dict) -> dict:
    """The map memory: for each kind of thing in view, where the nearest one is (absolute x, y)."""
    g, r, me = obs["grid"], len(obs["grid"]) // 2, obs["self"]
    best: dict = {}
    for j, row in enumerate(g):
        for i, ch in enumerate(row):
            if ch in "TMbvHQF*&^~" and (ch not in best or abs(i - r) + abs(j - r) < best[ch][0]):
                best[ch] = (abs(i - r) + abs(j - r), (me["x"] - r + i, me["y"] - r + j))
    for ch, (_, p) in best.items():
        seen[LEGEND.get(ch, ch).split(" (")[0].replace("_", " ")] = p
    return seen


def split_with(text: str) -> tuple[str, str | None]:
    w = text.split()
    if len(w) >= 2 and w[-2].lower() == "with":
        return " ".join(w[:-2]), w[-1]
    if w and w[-1].lower() in TOOLS and len(w) > 1:
        return " ".join(w[:-1]), w[-1]
    return text, None


def fail(error, detail):
    return {"ok": False, "error": error, "detail": detail}


def done(text, ticks=0):
    return {"ok": True, "result": {"ticks": ticks, "changed": {}, "text": text}}


class Verbs:
    async def verb(self, game, tool, args) -> dict:
        text = str(args.get("text", "")).strip()
        obs = await self.observe(game)
        if tool in ("where", "find"):
            spots = self.spots(obs, text)
            return done(f"{text}: " + (", ".join(f"({x},{y})" for x, y in spots[:4]) or "not in view and never seen"))
        if tool == "place_near":
            return await self.place_near(game, obs, text.split()[0] if text else "")
        if tool == "craft_here":
            w = text.split()
            name, n = (w[0] if w else ""), (int(w[1]) if len(w) > 1 and w[1].isdigit() else 1)
            st = RECIPES[name].station if name in RECIPES else None
            me = obs["self"]
            if st and not any(max(abs(p[0] - me["x"]), abs(p[1] - me["y"])) <= 2 for p in locate(obs, st)):
                spots = self.spots(obs, st)
                if spots:
                    r = await self.travel(game, obs, spots[0])
                    if not r.get("ok"):
                        return r
            return await game.act("craft", recipe=name, n=n)
        target, tool_in_hand = split_with(text) if tool in ("dig_at", "fight") else (text, None)
        dist = 8
        if tool == "move_away":
            w = target.split()
            num = [p.lstrip("-").isdigit() for p in w]
            if (len(w) == 3 and all(num)) or (len(w) >= 2 and num[-1] and not num[0]):
                dist, target = int(w[-1]), " ".join(w[:-1])
        spots = self.spots(obs, target)
        if not spots:
            return fail("not_found", f"no {target!r} in view, and none seen before")
        if tool == "move_away":
            return await self.move_away(game, obs, spots[0], dist)
        r = await self.travel(game, obs, spots[0])
        if not r.get("ok") or tool == "go_next_to":
            return r
        if tool == "fight":
            obs = await self.observe(game)
            me, name = obs["self"], target.lstrip("#").lower()
            e = min((e for e in obs["entities"] if name in (str(e["id"]), e["kind"], str(e.get("name", "")).lower())),
                    key=lambda e: abs(e["dx"]) + abs(e["dy"]), default=None)
            if not e:
                return fail("not_found", f"no {target!r} in view any more")
            return await game.act("attack", target=e["id"], **({"with": tool_in_hand} if tool_in_hand else {}))
        args = {"dir": r["result"]["dir"], **({"with": tool_in_hand} if tool_in_hand else {})}
        return await game.act("dig", **args)

    def spots(self, obs, target) -> list[tuple]:
        """Where target is: in view (nearest first), else where it was last seen (the player's map memory)."""
        here = locate(obs, target)
        if here:
            return here
        seen = getattr(self, "memory", {}).get("seen", {})
        key = target.lstrip("#").lower().replace("_", " ")
        return [tuple(seen[key])] if key in seen else []

    async def travel(self, game, obs, spot, legs=6) -> dict:
        """Next to spot, however far: walk legs toward it (pathfinding on each) until it is in view, then beside it."""
        for _ in range(legs):
            me = obs["self"]
            r = len(obs["grid"]) // 2
            if abs(spot[0] - me["x"]) <= r - 1 and abs(spot[1] - me["y"]) <= r - 1:
                return await self.go_next_to(game, obs, spot)
            reach = reachable(obs)
            q = min(reach, key=lambda q: math.dist((me["x"] + q[0], me["y"] + q[1]), spot))
            if q == (0, 0):
                res = await game.act("move_to", x=spot[0], y=spot[1])     # (the server's pathfinding: 64 tiles)
                if not res.get("ok"):
                    return fail("no_path", f"({spot[0]},{spot[1]}) cannot be reached from ({me['x']},{me['y']}): "
                                f"{res.get('detail') or res.get('error')}")
            else:
                res = await game.act("move_to", x=me["x"] + q[0], y=me["y"] + q[1], run=True)
                if not res.get("ok"):
                    return res
            obs = await self.observe(game)
        return fail("too_far", f"({spot[0]},{spot[1]}) still out of reach after {legs} legs")

    async def place_near(self, game, obs, item) -> dict:
        """Put item on a free, flat, placeable side: of you, else of the nearest reachable tile that has one."""
        last = None
        for d in free_dirs(obs):
            res = await game.act("place", dir=d, item=item)
            if res.get("ok") or res.get("error") in ("missing_item", "not_placeable"):
                return res
            last = res
        me = obs["self"]
        spots = sorted((n, q) for q, n in reachable(obs).items() if 0 < n <= 8 and free_dirs(obs, q))
        for n, q in spots[:3]:
            mv = await game.act("move_to", x=me["x"] + q[0], y=me["y"] + q[1])
            if not mv.get("ok"):
                continue
            obs = await self.observe(game)
            for d in free_dirs(obs):
                res = await game.act("place", dir=d, item=item)
                if res.get("ok") or res.get("error") in ("missing_item", "not_placeable"):
                    return res
                last = res
        return last or fail("no_room", "no free flat tile to put it on within 8 steps")

    async def go_next_to(self, game, obs, spot) -> dict:
        """Walk to the nearest reachable side of spot; result dir = the way it is from there."""
        me = obs["self"]
        reach = reachable(obs)
        sides = []
        for d, (dx, dy) in DIRS.items():
            sx, sy = spot[0] - dx, spot[1] - dy
            q = (sx - me["x"], sy - me["y"])
            sides.append((reach.get(q, 999), d, sx, sy))
        for steps, d, sx, sy in sorted(sides):
            if (sx, sy) == (me["x"], me["y"]):
                return {"ok": True, "result": {"ticks": 0, "changed": {}, "dir": d, "text": f"next to it: it is {d}"}}
            if steps == 999 and sides[0][0] != 999:
                break
            r = await game.act("move_to", x=sx, y=sy)
            if r.get("ok"):
                pos = (r.get("result") or {}).get("changed", {}).get("pos") or [sx, sy]
                if tuple(pos) == (sx, sy):
                    return {"ok": True, "result": {**r["result"], "dir": d, "text": f"next to ({spot[0]},{spot[1]}): it is {d}"}}
        return fail("no_path", f"no side of ({spot[0]},{spot[1]}) could be reached")

    async def move_away(self, game, obs, spot, dist) -> dict:
        """Walk to the reachable spot nearest to you that is at least dist from spot (else the farthest one)."""
        me = obs["self"]
        opts = [(math.dist((me["x"] + q[0], me["y"] + q[1]), spot), n, q) for q, n in reachable(obs).items()]
        if not opts:
            return fail("no_path", "nowhere reachable to go")
        ok = [o for o in opts if o[0] >= dist]
        best = min(ok, key=lambda o: o[1]) if ok else max(opts, key=lambda o: o[0])
        q = best[2]
        r = await game.act("move_to", x=me["x"] + q[0], y=me["y"] + q[1], run=True)
        if r.get("ok"):
            r.setdefault("result", {})["text"] = f"now {best[0]:.0f} tiles from ({spot[0]},{spot[1]})"
        return r

VERBS = set(VERB_SPECS)
