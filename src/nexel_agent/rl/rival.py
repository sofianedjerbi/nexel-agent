"""A hostile player in the sim (part of the environment, as a mob is: scripted, never the policy). Now and then it
turns up a few tiles off, walks up and swings (bare hands, sometimes a spear) for a while, as a hostile player does
live: it can kill, and a kill costs what it costs live - the pack falls where you die, and it takes it, then you wake
at your spawn. It comes back, soon after a win, later after a beating. Killed, it is gone till next time. It is
beatable: started at full health, a fight with bare hands goes your way more often than not. Sometimes it goes for a
neighbour near the player instead. The env is told how each visit ended."""

import math
import random

from nexel_worlds.sim.constants import sec

EVERY = (sec(100), sec(260))      # between two visits
AGAIN = (sec(60), sec(150))       # after it won (or you just took it): sooner
LAST = sec(12)                    # how long it keeps at it
SWING = sec(1.2)                  # (a blow a second or so)
HP = 6                            # (fists, a swing every 0.4 s: it falls in a few seconds of hitting back)
REACH, LOSE = 2.5, 7              # it swings from this close; this far, it has lost the player


class Rival:
    def __init__(self, sim, player, seed: int, pick=None):
        """pick(): whom it goes for this time (the player, or a neighbour near them); default the player."""
        self.sim, self.p, self.player = sim, player, player
        self.pick = pick or (lambda: player)
        self.rng = random.Random(seed * 31 + 7)
        self.a = sim.login("Rival", "rival-token")
        self.a.online = False
        self.next = sim.tick + self.rng.randint(*EVERY)
        self.until = self.swing = 0
        self.deaths, self.p_deaths, self.p_at = self.a.deaths, 0, (0, 0)
        self.outcome = None      # how the last visit ended: "won" (it was killed), "fled", "endured", "killed" (it killed)

    @property
    def on(self) -> bool:
        return self.a.online

    def dist(self) -> float:
        return math.dist((self.a.x, self.a.y), (self.p.x, self.p.y))

    def tick(self):
        sim, a, p = self.sim, self.a, self.p
        if not a.online:
            if sim.tick >= self.next and p.hp > 0:
                self.arrive()
            return
        if a.deaths > self.deaths:
            return self.leave("won", EVERY)
        if p.deaths > self.p_deaths:                       # it killed: the pack fell there, it takes it
            pile = sim.items.pop(self.p_at, None) or {}
            sim.dirty_items.add(self.p_at)
            for k, n in pile.items():
                a.add(k, n)
            return self.leave("killed", AGAIN)
        self.p_at = (p.x, p.y)
        d = self.dist()
        if d > LOSE:
            return self.leave("fled", EVERY)
        if sim.tick >= self.until:
            return self.leave("endured", AGAIN)
        if sim.tick < self.swing:
            return
        self.swing = sim.tick + SWING
        if d <= REACH:
            sim.submit(a, "attack", {"target": p.id, **({"with": "spear"} if a.count("spear") else {})})
        else:
            sim.submit(a, "move_to", {"x": p.x, "y": p.y})

    def arrive(self):
        self.p = self.pick()
        sim, a, p = self.sim, self.a, self.p
        for _ in range(30):
            ang, r = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(4, 6)
            x, y = p.x + round(r * math.cos(ang)), p.y + round(r * math.sin(ang))
            if sim.stand(x, y) and abs(sim.world.level(x, y) - sim.world.level(p.x, p.y)) <= 1:
                break
        else:
            self.next = sim.tick + sec(20)
            return
        a.inventory.clear()
        if self.rng.random() < 0.2:
            a.inventory["spear"] = 1
        a.x, a.y, a.z, a.hp, a.online = x, y, 99, HP, True
        self.deaths, self.p_deaths, self.p_at, self.outcome = a.deaths, p.deaths, (p.x, p.y), None
        self.until, self.swing = sim.tick + LAST, sim.tick + sec(1)

    def leave(self, outcome, wait):
        self.outcome = outcome
        self.sim.interrupt(self.a, "gone")
        self.a.online = False
        self.next = self.sim.tick + self.rng.randint(*wait)
