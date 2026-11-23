"""Goals: what the player is after right now, set from outside the policy (in the sim at random, so every kind gets
trained; live, by the LLM: rl/mind.py). The policy sees the goal (kind, item, the person and where they are) and is
paid once when it is done, read from the game: never per step of trying. Free play (no goal) is common."""

import math

from megastructure.sim.constants import sec

KINDS = ("free", "gather", "craft", "give", "ask", "stay", "help", "avoid")
WITH_PERSON = ("give", "ask", "stay", "help", "avoid")
GATHER = ("wood", "stone", "fiber", "berries", "dirt", "stone_block", "seed", "flower", "fruit")
CRAFT = ("plank", "workbench", "torch", "campfire", "axe", "pickaxe", "stone", "path", "door", "fence", "rope", "ladder", "stairs")
ASK = ("wood", "stone", "berries", "fiber", "plank")
REWARD = 1.0                    # once, when done
MAX_PER_EPISODE = 4             # goal rewards in one episode, at most
TIMEOUT = sec(180)
STAY_S, STAY_R, AVOID_R = sec(60), 4, 6
GATHER_N = 3


class Goal:
    def __init__(self, kind="free", item=None, person=None, start=0, base=0):
        self.kind, self.item, self.person, self.start, self.base = kind, item, person, start, base     # person: a name
        self.near_ticks, self.hit_foe, self.last = 0, False, start


class Goals:
    """In the sim: draws goals, tells when one is done (and pays it, a few times an episode at most)."""

    def __init__(self, sim, a, rng, peaceful=None):
        """peaceful(): the names of the people about who are not out to hurt the player (the neighbours): the goals
        with a person are drawn among them (help comes when one of them is set upon)."""
        self.sim, self.a, self.rng = sim, a, rng
        self.peaceful = peaceful or (lambda: [o.name for o in sim.agents.values() if o is not a])
        self.goal, self.paid, self.next = Goal(start=sim.tick), 0, sim.tick + sec(20)
        self.counts = {k: [0, 0] for k in KINDS if k != "free"}          # kind: [done, set]

    def agent(self, name):
        return self.sim.by_name.get(name) if name else None

    def draw(self, forced_person=None):
        sim, rng, a = self.sim, self.rng, self.a
        near = [n for n in self.peaceful() if (o := sim.by_name.get(n)) is not None and o.online
                and math.dist((o.x, o.y), (a.x, a.y)) <= 20]
        kinds = ["free"] * 4 + ["gather", "craft"] + (list(WITH_PERSON) if near else [])
        kind = "help" if forced_person else rng.choice(kinds)
        person = forced_person or (rng.choice(near) if kind in WITH_PERSON else None)
        has = sorted(sim.by_name[person].inventory) if kind == "ask" and person in sim.by_name else []
        pool = {"gather": GATHER, "craft": CRAFT, "ask": has or ASK}.get(kind)        # (ask for what they carry)
        item = rng.choice(pool) if pool else (max(a.inventory, key=a.inventory.get, default=None) if kind == "give" else None)
        self.goal = Goal(kind, item, person, sim.tick, a.count(item) if item else 0)
        if kind != "free":
            self.counts[kind][1] += 1
        self.next = sim.tick + TIMEOUT

    def step(self, results, got_from: dict, rival) -> float:
        """results: the player's actions that came back [(tool, args, result)]; got_from: {giver: {item: n}} since
        the last look; rival: the sparring partner (help). Returns the goal reward."""
        sim, a, g = self.sim, self.a, self.goal
        if sim.tick >= self.next:
            self.draw()
            return 0.0
        if g.kind == "free":
            return 0.0
        last, g.last = g.last, sim.tick
        p = self.agent(g.person)
        d = math.dist((p.x, p.y), (a.x, a.y)) if p is not None and p.online else 99.0
        done = False
        if g.kind == "gather":
            done = a.count(g.item) >= g.base + GATHER_N
        elif g.kind == "craft":
            done = any(t == "craft" and r.get("ok") and args.get("recipe") == g.item for t, args, r in results)
        elif g.kind == "give":
            done = any(t == "give" and r.get("ok") and r.get("to") == g.person for t, args, r in results)
        elif g.kind == "ask":
            done = g.item in got_from.get(g.person, {})
        elif g.kind == "stay":
            g.near_ticks += (sim.tick - last) if d <= STAY_R else 0      # (time spent near them, all told)
            done = g.near_ticks >= STAY_S
        elif g.kind == "avoid":
            if d <= AVOID_R:
                g.start = sim.tick                          # (came close: the minute starts again)
            done = sim.tick - g.start >= STAY_S
        elif g.kind == "help" and rival is not None:
            g.hit_foe |= any(t == "attack" and r.get("ok") and args.get("target") == rival.a.id for t, args, r in results)
            done = g.hit_foe and rival.outcome is not None
        if not done:
            return 0.0
        self.counts[g.kind][0] += 1
        self.goal, self.next = Goal(start=sim.tick), sim.tick + sec(10)      # (a spell of free play, then another)
        if self.paid >= MAX_PER_EPISODE:
            return 0.0
        self.paid += 1
        return REWARD
