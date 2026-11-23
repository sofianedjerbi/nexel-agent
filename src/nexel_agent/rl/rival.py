"""A sparring partner in the sim: another person who now and then turns up near the player and goes for it (part of
the environment, as a mob is: scripted, never the policy). It comes from a few tiles off, walks up, swings (bare
hands, sometimes a spear), and gives up after a while or when the player has got away; killed, it is gone till next
time. Sometimes it goes for a neighbour near the player instead. The player's side of it is told to the env: who hit,
whether it got away, whether it won."""

import math
import random

from megastructure.sim.constants import sec

EVERY = (sec(100), sec(260))      # between two visits
LAST = sec(8)                     # how long it keeps at it
SWING = sec(1.2)                  # (a blow a second or so: a third of your health if you just stand there)
HP = 6                            # (fists: a few seconds of hitting back see it off)
REACH, LOSE = 2.5, 7              # it swings from this close; this far, it has lost the player
SPARE = 4                         # it stops at this much health left (half hearts)


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
        self.deaths = self.a.deaths
        self.outcome = None               # what the last visit came to: "won" (player killed it), "fled", "endured"

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
            return self.leave("won")
        d = self.dist()
        if d > LOSE:
            return self.leave("fled")
        if sim.tick >= self.until or p.hp <= SPARE:          # (a sparring partner: it leaves you a little life)
            return self.leave("endured")
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
            if sim.world.passable(x, y) and abs(sim.world.level(x, y) - sim.world.level(p.x, p.y)) <= 1:
                break
        else:
            self.next = sim.tick + sec(20)
            return
        a.inventory.clear()
        if self.rng.random() < 0.3:
            a.inventory["spear"] = 1
        a.x, a.y, a.z, a.hp, a.online = x, y, 99, HP, True
        self.deaths, self.outcome = a.deaths, None
        self.until, self.swing = sim.tick + LAST, sim.tick + sec(1)

    def leave(self, outcome):
        self.outcome = outcome
        self.sim.interrupt(self.a, "gone")
        self.a.online = False
        self.next = self.sim.tick + self.rng.randint(*EVERY)
