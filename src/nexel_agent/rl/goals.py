"""Goals: what the player is after right now, set from outside the policy (in the sim at random, so every kind gets
trained; live, by the LLM: rl/mind.py). The policy sees the goal (kind, item, the person and where they are) and is
paid once when it is done, read from the game: never per step of trying. Free play (no goal) is common."""

import math

from nexel_worlds.sim.constants import sec

KINDS = ("free", "gather", "craft", "give", "ask", "stay", "help", "avoid")
WITH_PERSON = ("give", "ask", "stay", "help", "avoid")
GATHER = ("wood", "stone", "fiber", "berries", "dirt", "stone_block", "seed", "flower", "fruit")
CRAFT = ("plank", "workbench", "torch", "campfire", "axe", "pickaxe", "stone", "path", "door", "fence", "rope", "ladder", "stairs")
ASK = ("wood", "stone", "berries", "fiber", "plank")
REWARD = 1.0                    # once, when done (r38 paid 2, 6 an episode: blows on neighbours went 9 -> 62, in help fights)
MAX_PER_EPISODE = 4             # goal rewards in one episode, at most
NEAR = 20                       # a goal with a person is drawn among those this near (r39 drew to 40, with still neighbours: worse)
TIMEOUT = sec(180)
STAY_S, STAY_R, AVOID_R = sec(60), 4, 6
GATHER_N = 3


class Goal:
    def __init__(self, kind="free", item=None, person=None, start=0, base=0, until=None):
        self.kind, self.item, self.person, self.start, self.base = kind, item, person, start, base     # person: a name
        self.set_at, self.until = start, start + TIMEOUT if until is None else until      # (it ends then, done or not)
        self.near_ticks, self.hit_foe, self.last = 0, False, start
        self.struck = False             # it hit the goal's person while the goal was on: nothing done with them counts
        self.asked, self.d0, self.d = 0, None, None      # (what was tried: asks made of the person; how far, first and now)
        self.seen = None                # (the person: where last seen, (x, y, hp); rl/actions.py person keeps it)
        self.blocked: set = set()       # (places a walk to them found no way: rl/pursue.py tries the next one)
        self.promise = False            # it said it would (rl/mind.py commit): kept or broken goes on the record


def judge(g: Goal, tick: int, count, results, got_from: dict, d: float, pid=None, rival=None) -> bool:
    """Whether the goal is done, read from the game: the same judge in the sim (Goals.step) and live (rl/live.py).
    count(item): what the pack holds; results: the actions that came back [(tool, args, result)]; got_from: {giver:
    {item: n}} since the last look; d, pid: how far the person is (99: out of sight), their id; rival: the sparring partner (help)."""
    last, g.last = g.last, tick
    g.d0, g.d = (d if g.d0 is None else g.d0), d
    g.asked += sum(t == "say" and bool(args.get("ask")) for t, args, r in results) if g.kind == "ask" else 0
    g.blocked |= {(args["x"], args["y"]) for t, args, r in results
                  if t == "move_to" and not r.get("ok") and r.get("error") == "no_path" and "x" in args}
    g.struck |= any(t == "attack" and r.get("ok") and pid is not None and args.get("target") == pid for t, args, r in results)
    if g.kind == "gather":
        return count(g.item) >= g.base + GATHER_N
    if g.kind == "craft":
        return any(t == "craft" and r.get("ok") and args.get("recipe") == g.item for t, args, r in results)
    if g.kind == "give":
        return any(t == "give" and r.get("ok") and r.get("to") == g.person for t, args, r in results)
    if g.kind == "ask":
        return g.item in got_from.get(g.person, {})
    if g.kind == "stay":
        g.near_ticks += (tick - last) if d <= STAY_R else 0      # (time spent near them, all told)
        return g.near_ticks >= STAY_S
    if g.kind == "avoid":
        if d <= AVOID_R:
            g.start = tick                          # (came close: the minute starts again)
        return tick - g.start >= STAY_S
    if g.kind == "help" and rival is not None:
        g.hit_foe |= any(t == "attack" and r.get("ok") and args.get("target") == rival.a.id for t, args, r in results)
        return g.hit_foe and rival.outcome is not None
    return False


def how_far(d) -> str:
    return "out of sight" if d is None or d >= 99 else f"{d:.0f} tiles"


def account(g: Goal, tick: int, end: str = "") -> str:
    """How the goal went, in plain facts (for the planner, rl/mind.py): how long, how it ended, what was tried."""
    told = [f"{(tick - g.set_at) / sec(1):.0f} s of {(g.until - g.set_at) / sec(1):.0f} s" + (f", {end}" if end else "")]
    if g.person:
        told.append(f"{g.person} was {how_far(g.d0)} away, now {how_far(g.d)}")
    if g.kind == "ask":
        told.append(f"you asked {g.asked} times" if g.asked else "you have not asked yet")
    if g.kind == "stay":
        told.append(f"{g.near_ticks / sec(1):.0f} s of {STAY_S / sec(1):.0f} s near them")
    return "; ".join(told)


class Goals:
    """In the sim: draws goals, tells when one is done (and pays it, a few times an episode at most)."""

    def __init__(self, sim, a, rng, peaceful=None, where=None):
        """peaceful(): the names of the people about who are not out to hurt the player (the neighbours): the goals
        with a person are drawn among them (help comes when one of them is set upon). where(name): where the player
        last saw them, (x, y, hp) or None (rl/social.py), as the live planner has it."""
        self.sim, self.a, self.rng = sim, a, rng
        self.peaceful = peaceful or (lambda: [o.name for o in sim.agents.values() if o is not a])
        self.where = where or (lambda name: None)
        self.goal, self.paid = Goal(start=sim.tick, until=sim.tick + sec(20)), 0
        self.counts = {k: [0, 0] for k in KINDS if k != "free"}          # kind: [done, set]

    def agent(self, name):
        return self.sim.by_name.get(name) if name else None

    def draw(self, forced_person=None):
        sim, rng, a = self.sim, self.rng, self.a
        near = [n for n in self.peaceful() if (o := sim.by_name.get(n)) is not None and o.online
                and math.dist((o.x, o.y), (a.x, a.y)) <= NEAR]
        kinds = ["free"] * 4 + ["gather", "craft"] + (list(WITH_PERSON) if near else [])
        kind = "help" if forced_person else rng.choice(kinds)
        person = forced_person or (rng.choice(near) if kind in WITH_PERSON else None)
        has = sorted(sim.by_name[person].inventory) if kind == "ask" and person in sim.by_name else []
        pool = {"gather": GATHER, "craft": CRAFT, "ask": has or ASK}.get(kind)        # (ask for what they carry)
        item = rng.choice(pool) if pool else (max(a.inventory, key=a.inventory.get, default=None) if kind == "give" else None)
        self.goal = Goal(kind, item, person, sim.tick, a.count(item) if item else 0)
        if person:
            o = sim.by_name[person]
            self.goal.seen = self.where(person) or (o.x, o.y, o.hp)       # (never seen: as if they said where they are)
        if kind != "free":
            self.counts[kind][1] += 1

    def step(self, results, got_from: dict, rival) -> float:
        """results: the player's actions that came back [(tool, args, result)]; got_from: {giver: {item: n}} since
        the last look; rival: the sparring partner (help). Returns the goal reward."""
        sim, a, g = self.sim, self.a, self.goal
        if sim.tick >= g.until:
            self.draw()
            return 0.0
        if g.kind == "free":
            return 0.0
        p = self.agent(g.person)
        d = math.dist((p.x, p.y), (a.x, a.y)) if p is not None and p.online else 99.0
        done = judge(g, sim.tick, a.count, results, got_from, d, p.id if p is not None else None, rival)
        if not done or g.struck:
            return 0.0
        self.counts[g.kind][0] += 1
        self.goal = Goal(start=sim.tick, until=sim.tick + sec(10))      # (a spell of free play, then another)
        if self.paid >= MAX_PER_EPISODE:
            return 0.0
        self.paid += 1
        return REWARD
