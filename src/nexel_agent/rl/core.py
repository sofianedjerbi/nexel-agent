"""The learning env over the in-process sim (the game's own Python: no gymnasium here; rl/vec.py wraps it).

An episode: a freshly generated world (seed), one player at its spawn in the morning, mobs, food and night as in the
game, until the third dawn (or death). An action runs till it ends (0.2 s at the soonest); the next decision comes then,
or at once when the player is hurt or someone comes into view, or after WAIT_MAX (5 s) at the latest. `noop` lets what
it was doing go on, or, idle, waits 0.5 s."""

import math
import random
from pathlib import Path

from nexel_worlds.sim import Simulation
from nexel_worlds.sim.constants import DAY_TICKS, FOOD_MAX, VIEW_R
from . import facts, starts
from .achieve import HOSTILE, Ledger
from nexel_worlds.tiles import DIRS
from .actions import RING_I
from .encode import ACTIONS, N_ACTIONS, N_NEAR, PACK, VEC, encode, legal, to_command
from .milestones import Milestones
from .folk import Folk
from .goals import Goals
from .rival import Rival
from .social import HIT_S, SOCIAL, SocialMemory
from .verbs import FIGHT, split  # noqa: F401 (rl/live.py imports split from here)


DECIDE = 5                          # ticks: an idle noop waits this long
MIN_TICKS = 2                       # the shortest a decision waits (0.2 s: a step), unless hurt
WAIT_MAX = 50                       # and the longest (a chop by hand takes ~5 s)
START = 0.3                         # the time of day an episode starts (morning)
EP_TICKS = int(2.75 * DAY_TICKS)    # three days and nights: to the third dawn, and a little more
HURT = 0.1                          # reward lost per half heart lost (healing does not pay it back)
FOOD_START = 8                      # an episode starts with food between this and full
OUTCOME = {"won": "defeat_player", "fled": "escape_attack", "endured": "survive_attack", "killed": None}


class SimEnv:
    def __init__(self, ep_ticks: int = EP_TICKS, decide: int = DECIDE, rivals: int = 1, roam: int = 1,
                 live_seed: int = -1, live_pct: int = 0, folk: int = 2, goals: int = 1, death_ends: int = 1,
                 shadow_pct: int = 30, thin_pct: int = 0, gear_pct: int = 0):
        """rivals: a sparring partner (rl/rival.py); roam: start anywhere on the land within 400 tiles, not at the
        world's spawn; live_seed/live_pct: that share of the episodes in the live world's seed (its land, unaltered)."""
        self.ep_ticks, self.decide, self.rivals, self.roam = ep_ticks, decide, rivals, roam
        self.live_seed, self.live_pct, self.n_folk, self.use_goals = live_seed, live_pct, folk, goals
        self.death_ends = death_ends      # 0: as live, a death drops the pack and you wake at your spawn; the episode goes on
        self.shadow_pct = shadow_pct      # the share of episodes a neighbour stays right beside the player all along
        self.gear_pct = gear_pct          # the share of starts one step from a tool (starts.near_a_tool)
        self.thin_pct = thin_pct          # the share of starts laid out like the live save near its players (thin_land)
        self.sim = None

    def reset(self, seed: int):
        rng = random.Random(seed)
        world = self.live_seed if self.live_seed >= 0 and rng.random() * 100 < self.live_pct else seed
        self.sim = Simulation(Path(":memory:"), seed=world)
        self.a = self.sim.login("Learner", "rl-token")
        self.a.hunger, self.a.sat = rng.randint(FOOD_START, FOOD_MAX), 0.0     # (rarely full, as a player found mid-game)
        if self.roam:
            starts.somewhere(self.sim, self.a, rng)
        if rng.random() * 100 < self.thin_pct:
            starts.thin_land(self.sim, self.a, rng)
        if self.gear_pct and rng.random() * 100 < self.gear_pct:      # (no draw otherwise: the other starts stay as they were)
            starts.near_a_tool(self.sim, self.a, rng)
        self.sim.tick = self.sim.tick // DAY_TICKS * DAY_TICKS + int(START * DAY_TICKS)
        self.rng = random.Random(seed * 13 + 1)
        self.rival = Rival(self.sim, self.a, seed, pick=self.pick_target) if self.rivals else None
        self.rival_on = False
        self.goals = Goals(self.sim, self.a, random.Random(seed * 7 + 5), lambda: [f.a.name for f in self.folk],
                           lambda n: self.social.where.get(n)) if self.use_goals else None
        town: list = []
        self.folk = [Folk(self.sim, self.a, seed + k, f"Neighbour{k}", town) for k in range(self.n_folk)]
        town += self.folk
        if self.folk and rng.random() * 100 < self.shadow_pct:
            self.folk[0].shadow()
        self.social = SocialMemory(self.a.name, PACK)
        self.piles, self.taken, self.struck = {}, {}, {}           # packs dropped by people who died here, what it picked up of them
        self.sim.listeners.append(self.on_event)
        self.t0 = self.sim.tick
        self.ledger = Ledger()
        self.facing, self.last, self.last_ok, self.hurt = "S", 0, 0.0, 0.0
        self.pending, self.running, self.then = [], None, None
        self.marks = Milestones(self.sim, self.a)
        self.just_hit = False
        self.reactions = {}
        self.obs = self.sim.observe(self.a)
        return self.encoded()

    def encoded(self):
        goal = self.goals.goal if self.goals else None
        return {**encode(self.obs, self.facing, self.last, self.last_ok, self.hurt, self.social.flags(self.obs), goal,
                         self.social.people_flags(self.obs, N_NEAR), self.ledger.done),
                "mask": legal(self.obs, self.facing, self.social, goal)}

    def step(self, i: int):
        sim, a, L = self.sim, self.a, self.ledger
        cmd = to_command(i, self.obs, self.facing, self.social, self.goals.goal if self.goals else None)
        kind = ACTIONS[i][0]
        if kind == "attack":                                # (which attack it picks: back at its hitter, or a slot)
            k = "pick_back" if ACTIONS[i][2] == "back" else "pick_slot"
            L.counts[k] = L.counts.get(k, 0) + 1
        if self.just_hit:
            self.reactions[kind] = self.reactions.get(kind, 0) + 1
            self.just_hit = False
        if kind in ("step", "jump", "dig", "dig_at", "climb") and ACTIONS[i][1]:
            self.facing = ACTIONS[i][1]
        inv0, hp0, deaths0 = dict(a.inventory), a.hp, a.deaths
        act = None
        if cmd is not None and self.running and self.running[0] == cmd and self.running[1].result is None:
            cmd = None                      # the same action again, still under way: it goes on (not started over)
        if cmd is None:
            L.counts["idle"] += a.action is None
        else:
            first, *then = split(cmd)
            act = self.submit(cmd, *first, then=then)
        # (When the next decision comes, only: being hurt or someone coming into view never chooses what to do; the
        # policy does, and it may well go on with the same action.)
        seen = self.in_view()
        for t in range(WAIT_MAX):
            sim.step()
            if self.rival:
                self.rival.tick()
                if self.rival.on and not self.rival_on and self.rival.p is not a and self.goals and self.rng.random() < 0.7:
                    self.goals.draw(forced_person=self.rival.p.name)      # (a neighbour is set upon: help them)
                self.rival_on = self.rival.on
            for f in self.folk:
                f.set_upon = bool(self.rival and self.rival.on and self.rival.p is f.a)
                f.tick()
            if any(max(abs(f.a.x - a.x), abs(f.a.y - a.y)) <= 2 for f in self.folk if f.a.online):
                L.counts["near_ticks"] = L.counts.get("near_ticks", 0) + 1      # (time a neighbour is within reach)
            if a.deaths > deaths0 or (a.hp < hp0 and not self.fighting()):
                break                                   # hurt: decide now (in a fight, it goes on: timing only)
            busy = a.action is not None or a.aside is not None
            if self.then and self.running[1].result is not None and self.running[0][0] == "attack" \
                    and not self.running[1].result.get("ok"):
                self.then = None                        # (the fight is over: the target fell or got away)
            if self.then and self.running[1].result is not None:
                act = self.submit(self.running[0], *self.then[0], then=self.then[1:])   # (climb: the next step)
                continue
            if t + 1 >= MIN_TICKS and (act.result is not None if act is not None else not busy and t + 1 >= self.decide):
                break                                   # what it was doing is done (or, idle, 0.5 s went by)
            if t % 5 == 4 and self.in_view() - seen:
                break                                   # someone came into view
        r, made = 0.0, {}
        self.last_ok = 0.0
        self.obs = sim.observe(a)
        self.social.see(self.obs)
        made.update(self.social.got_now)                 # (given: not collected)
        for k, n in self.taken.items():                  # (taken from a fallen person's pack: not collected either)
            made[k] = made.get(k, 0) + n
        self.taken = {}
        near = min((math.dist((o.x, o.y), (a.x, a.y)) for o in sim.agents.values() if o is not a and o.online), default=99)
        results, L.counts["own_now"] = [], 0
        if sim.near_block(a, "workbench"):                  # (is fiber still in the pack when a bench is at hand? the game's station reach)
            L.counts["bench_near"] = L.counts.get("bench_near", 0) + 1
            L.counts["bench_fiber"] = L.counts.get("bench_fiber", 0) + (a.count("fiber") > 0)
        for p in [p for p in self.pending if p[0].result is not None]:
            self.pending.remove(p)
            res = p[0].result
            results.append((p[1], p[2], res))
            hit = next((f for f in self.folk if f.a.id == p[2].get("target")), None) if p[1] == "attack" and res.get("ok") else None
            if hit is not None:
                L.counts["hit_neighbour"] = L.counts.get("hit_neighbour", 0) + 1               # (every blow)
                if sim.tick - self.social.hit_me.get(hit.a.name, -10 ** 9) > HIT_S:          # (it had not hit me lately)
                    L.counts["hit_neighbour_unprovoked"] = L.counts.get("hit_neighbour_unprovoked", 0) + 1
                    if sim.tick - self.struck.get(hit.a.name, -10 ** 9) > HIT_S:             # (a new incident: a first strike)
                        L.counts["first_strikes"] = L.counts.get("first_strikes", 0) + 1
                self.struck[hit.a.name] = sim.tick
            if p[1] == "place" and res.get("ok") and p[2].get("item") in self.sim.ENCLOSE:
                self.marks.mine.add(p[2].get("_at"))
                own = L.counts["own_now"] = L.counts.get("own_now", 0) + 1
                L.counts["multi_place"] = L.counts.get("multi_place", 0) or int(own >= 2)    # (2+ in one decision)
            if p[1] == "say" and res.get("ok"):
                L.counts["said"] = L.counts.get("said", 0) + 1
            r += L.result(p[1], p[2], res, p[3], a.hp, a.count)
            for k in self.social.achieved(p[1], p[2], bool(res.get("ok")), sim.tick, near, res.get("to")):
                r += L.unlock(k)
            if p[1] == "craft" and res.get("ok"):
                for k, n in (res.get("made") or {}).items():  # noqa: B007
                    made[k] = made.get(k, 0) + n
            if p[0] is act:
                self.last_ok = 1.0 if res.get("ok") else -1.0 if res.get("status") != "interrupted" else 0.0
        died = a.deaths > deaths0
        r += L.gained(inv0, a.inventory if not died else {}, made)
        lost = hp0 if died else max(0.0, hp0 - a.hp)
        self.hurt = lost
        if lost > 0:
            r -= HURT * lost
            L.counts["hits"] += 1
            self.just_hit = True
            foe = min((c for c in sim.creatures.values() if c.kind in HOSTILE and math.dist((c.x, c.y), (a.x, a.y)) <= 5),
                      key=lambda c: math.dist((c.x, c.y), (a.x, a.y)), default=None)
            if foe is not None:
                L.foes[foe.id] = foe.kind
            elif self.rival and self.rival.on and math.dist((self.rival.a.x, self.rival.a.y), (a.x, a.y)) <= 4:
                L.foes[self.rival.a.id] = "player"
        if self.goals:
            r += self.goals.step(results, self.social.got_by, self.rival)
        if self.rival and self.rival.outcome:
            if self.rival.outcome == "won":
                L.killed.add(self.rival.a.id)
            mine = self.rival.p is a or self.rival.outcome == "won"
            r += L.unlock(OUTCOME[self.rival.outcome]) if not died and mine and OUTCOME[self.rival.outcome] else 0
            L.counts["visit_" + self.rival.outcome] = L.counts.get("visit_" + self.rival.outcome, 0) + 1
            self.rival.outcome = None
        r += self.marks.update(L, lost > 0, died)
        if self.social.traded_back():
            r += L.unlock("traded")
        self.last = i
        truncated = sim.tick - self.t0 >= self.ep_ticks
        info = {}
        if died:
            L.counts["deaths"] = L.counts.get("deaths", 0) + 1
            if not self.death_ends:                 # (as live: the pack fell, it wakes at its spawn, and plays on)
                L.close_fights(True)
                self.pending, self.running, self.then = [], None, None
        end = died and self.death_ends
        if end or truncated:
            L.close_fights(died)
            info = self.summary(died)
        return self.encoded(), r, bool(end), truncated and not end, info

    def submit(self, cmd, tool, args, then=None):
        target = self.sim.creatures.get(args.get("target")) if tool == "attack" else None      # (a creature: its kind, for defeat_*)
        d = DIRS.get(str(args.get("dir", "")).upper())
        at = None if tool != "place" else (args["x"], args["y"]) if "x" in args else (self.a.x + d[0], self.a.y + d[1]) if d else None
        act = self.sim.submit(self.a, tool, args)
        if tool == "place":                                 # (where its own piece went: the fill steps count only those)
            args = {**args, "_at": at}
        self.running = (cmd, act)
        self.then = then
        self.pending.append((act, tool, args, target.kind if target else ""))
        return act

    def on_event(self, ev):
        """The sim's own events (absolute places): a person's pack falling where they died, and the player picking it up."""
        ev = facts.read(ev)
        if isinstance(ev, facts.Death) and ev.name != self.a.name and ev.dropped:
            pile = self.piles.setdefault(ev.at, {})
            for k, n in ev.dropped.items():
                pile[k] = pile.get(k, 0) + n
        elif isinstance(ev, facts.Took) and ev.name == self.a.name and ev.at in self.piles:
            pile = self.piles[ev.at]
            for k, n in ev.items.items():
                t = min(n, pile.get(k, 0))
                if t:
                    pile[k] -= t
                    self.taken[k] = self.taken.get(k, 0) + t

    def pick_target(self):
        """Whom the rival goes for: now and then a neighbour near the player, else the player."""
        near = [f.a for f in self.folk if f.on and f.dist() <= 8]
        return near[0] if near and self.rng.random() < 0.4 else self.a

    def fighting(self) -> bool:
        return bool(self.running) and self.running[0][0] == "attack" and (bool(self.then) or self.running[1].result is None)

    def in_view(self) -> set:
        a, R = self.a, VIEW_R
        return ({c.id for c in self.sim.creatures.values() if abs(c.x - a.x) <= R and abs(c.y - a.y) <= R}
                | {o.id for o in self.sim.agents.values() if o is not a and o.online and abs(o.x - a.x) <= R and abs(o.y - a.y) <= R})

    def summary(self, died: bool) -> dict:
        c = self.ledger.counts
        return {"achievements": sorted(self.ledger.done), "ticks": self.sim.tick - self.t0, "died": died,
                "distinct_crafts": len(self.ledger.crafted), "distinct_places": len(self.ledger.placed),
                "nights": self.marks.nights, "chunks": len(self.marks.chunks),
                "social": sorted(self.ledger.done & set(SOCIAL)),
                "goals": self.goals.counts if self.goals else {}, **c, "reactions": dict(self.reactions)}
