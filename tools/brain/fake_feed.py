"""A made-up player on a made-up patch of land, served through the real feed (rl/brain.py), to work on the brain view
with no game and no model running:
    python tools/brain/fake_feed.py [--port 8790] [--seconds 600]
    open tools/brain/index.html?feed=http://127.0.0.1:8790/feed"""

import argparse
import random
import time

from nexel_agent.rl.brain import Brain, chat_or_event, frame, top
from nexel_agent.rl.goals import Goal
from nexel_agent.rl.judge import CHECKS, LOOKS, SENSE

ACTS = ["step_N", "step_E", "step_S", "step_W", "dig_W", "attack", "approach", "say", "eat_berries", "craft_plank", "place_dirt_N"]


def land(rng, w=60):
    rows = []
    for y in range(w):
        r = ""
        for x in range(w):
            v = rng.random()
            r += ("~" if x > 44 and y < 16 else "," if 20 < x < 26 and 30 < y < 36 else "_" if y == 28 and x < 40 else
                  "T" if v < .1 else "b" if v < .12 else "f" if v < .15 else "M" if v < .17 else ".")
        rows.append(r)
    for x, y, ch in ((30, 26, "&"), (31, 26, "C"), (29, 25, "#"), (30, 25, "#"), (31, 25, "#"), (32, 25, "+"), (28, 27, "^")):
        rows[y] = rows[y][:x] + ch + rows[y][x + 1:]
    return rows


class Fake:
    def __init__(self, seed=3):
        self.rng = random.Random(seed)
        self.map = land(self.rng)
        self.me, self.mira, self.spore = [30, 30], [33, 28], [24, 31]
        self.hp, self.food, self.facing = 17.0, 15.0, "S"
        self.inv = {"wood": 4, "stone_block": 12, "dirt": 3}
        self.goal, self.goal_at = Goal("gather", "wood", "Mira"), time.monotonic()
        self.relations = {"Mira": {"ally": True, "notes": "shares seeds, said she'd help with the house"}}
        self.foes, self.hurt, self.pol, self.mind = {}, 0.0, None, None
        self.tried = {"gather wood (with Mira)": [2, 1], "make a bed": [1, 0]}   # (as mind.tried: goal: [set, done])
        self.home, self.home_read = None, None                 # (as the mind's: rl/home.py reading, made up here)

    def obs(self, events=()):
        x0, y0 = self.me
        grid = []
        for y in range(y0 - 10, y0 + 11):
            grid.append("".join(self.map[y % 60][x % 60] for x in range(x0 - 10, x0 + 11)))
        put = lambda g, dx, dy, ch: g.__setitem__(10 + dy, g[10 + dy][:10 + dx] + ch + g[10 + dy][11 + dx:])
        ents = []
        for p, kind, name, hp, ch in ((self.mira, "agent", "Mira", 14, "P"), (self.spore, "sporeling", None, 2, "s")):
            dx, dy = p[0] - x0, p[1] - y0
            if abs(dx) <= 10 and abs(dy) <= 10:
                put(grid, dx, dy, ch)
                ents.append({"id": hash(kind), "kind": kind, "dx": dx, "dy": dy, "hp": hp, **({"name": name} if name else {})})
        put(grid, 0, 0, "@")
        ents.sort(key=lambda e: abs(e["dx"]) + abs(e["dy"]))
        hh = 17 + int(time.time() / 4) % 7
        return {"tick": 1, "time": f"day 3, {hh}:{int(time.time()) % 60:02d}" + (" (night)" if hh >= 18 else ""),
                "self": {"x": x0, "y": y0, "z": 0, "hp": self.hp, "hunger": self.food, "deaths": 0, "inventory": dict(self.inv)},
                "grid": grid, "entities": ents, "events": list(events)}

    def step(self, rng):
        d = rng.choice("NESW")
        self.facing = d
        self.me[0] += {"E": 1, "W": -1}.get(d, 0) * (rng.random() < .6)
        self.me[1] += {"S": 1, "N": -1}.get(d, 0) * (rng.random() < .6)
        for p in (self.mira, self.spore):
            p[0] += rng.choice((-1, 0, 1)) if abs(p[0] - self.me[0]) < 6 else (1 if p[0] < self.me[0] else -1)
            p[1] += rng.choice((-1, 0, 1)) if abs(p[1] - self.me[1]) < 6 else (1 if p[1] < self.me[1] else -1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", type=int, default=8790)
    a.add_argument("--seconds", type=float, default=600)
    a = a.parse_args()
    f = Fake()
    b = Brain(a.port, None, "Rook", "easygoing builder, a bit stubborn")
    print(f"feed on http://127.0.0.1:{b.port}/feed")
    b.emit({"k": "memory", "people": f.relations})
    rng, t_end, n = f.rng, time.monotonic() + a.seconds, 0
    while time.monotonic() < t_end:
        f.step(rng)
        events = []
        if rng.random() < .15:
            events.append({"type": "attack", "attacker": "a sporeling", "target": "Rook", "damage": 1,
                           "who": {"kind": "sporeling", "id": 9, "name": None}, "whom": {"kind": "agent", "id": 1, "name": "Rook"}})
            f.hp = max(4.0, f.hp - 1)
        if rng.random() < .08:
            events.append({"type": "say", "from": "Mira", "text": rng.choice(["got any stone?", "nice house", "brute's back", "ty rook"])})
        o = f.obs(events)
        logits = [rng.gauss(0, 1.2) for _ in range(184)]
        mask = [rng.random() < .25 for _ in range(184)]
        for i in range(len(ACTS)):
            mask[i] = True
            logits[i] += 2.5 if i in (4, 5) else 1
        names = lambda i: ACTS[i] if i < len(ACTS) else f"act_{i}"
        f_ = frame(o, f.facing, rng.choice(ACTS[:6]), mask, f.goal, time.monotonic() - f.goal_at, {"Mira"}, set())
        f_["top"] = top(logits, mask, names)
        f_["value"] = round(.4 + .3 * rng.random(), 3)
        for e in events:
            line = chat_or_event(e, "Rook")
            if line:
                b.emit(line)
        b.emit(f_)
        n += 1
        if n % 12 == 3:
            facts = (f"You: health {f.hp:g}/20, food {f.food:g}/20, carrying wood {f.inv['wood']}, stone_block 12, dirt 3. {o['time']}.\n"
                     f"Your goal now: gather wood (with Mira), set {n} s ago.\n"
                     f"Mira: 3 tiles away, health 14, holding hoe. Your notes: {f.relations['Mira']['notes']} (your ally). "
                     f"Lately: {'asked you for stone' if n % 24 == 3 else 'nothing'}.")
            say = rng.choice(["hang on mira", "wood's close", "", "on it"])
            b.emit({"k": "thought", "facts": facts, "took_s": round(.6 + rng.random(), 2),
                    "reply": {"goal": "gather", "with": "Mira", "item": "wood", "say": say,
                              "relation_update": {"Mira": {"ally": True, "note": "still helping with the house"}}}})
            if say:
                b.emit({"k": "chat", "from": "Rook", "text": say, "said": True})
            f.inv["wood"] += 1
        if n % 30 == 20:                                      # (a goal ends now and then: the tally moves)
            g = rng.choice(list(f.tried) + ["ask Mira for stone"])
            t = f.tried.setdefault(g, [0, 0])
            t[0] += 1
            t[1] += rng.random() < .6
        b.tried(f)
        if n == 15:                                           # (it chooses a spot: not closed yet, lacking the hut's wood)
            f.home = (31, 26)
            f.home_read = {"at": [31, 26], "where": "4 tiles N", "far": False, "closed": False, "told": "not closed",
                           "verdict": {"grade": 0, "of": 6, "checks": dict.fromkeys(CHECKS, False), "failed": list(CHECKS)},
                           "lacks": {"wood": 14, "fiber": 3}}
        if n == 45:                                           # (walls up and a door: closed, the rest missing)
            f.home_read = {**f.home_read, "closed": True, "told": "closed, a door, no roof (0% covered), no light, no bed, 9 tiles inside (grade 3 of 6)",
                           "verdict": {"grade": 3, "of": 6, "checks": {**dict.fromkeys(CHECKS, False), "closed": True, "door": True, "room_enough": True},
                                       "failed": ["roofed", "lit", "bed"]},
                           "sense": {"grade": 3, "of": len(SENSE), "failed": ["chest", "station", "seat", "fit", "light_by_bed"]},
                           "looks": {"grade": 4, "of": len(LOOKS), "failed": ["roof", "path", "no_stray"]},
                           "lacks": {"roof": 9, "bed": 1}}
        b.homed(f)
        time.sleep(1.0)
    b.close()


if __name__ == "__main__":
    main()
