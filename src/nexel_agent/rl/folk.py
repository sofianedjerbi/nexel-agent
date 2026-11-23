"""Peaceful neighbours in the sim (part of the environment, as the rival is: scripted, never the policy). Now and
then one comes by and spends a few minutes around the player: wanders near, works (chops, digs), and once in a
while says hello, asks for something, or hands something over; given something, it sometimes gives back. They say
plain short lines; on the live server real players and the LLM's chat take their place."""

import math
import random

from megastructure.sim.constants import sec
from megastructure.tiles import DIRS

EVERY = (sec(60), sec(200))          # between two visits
STAY = (sec(120), sec(300))
ACT = (sec(3), sec(8))               # between two things it does
ASKS = ("wood", "stone", "berries", "fiber", "plank", "dirt")
GIFTS = {"berries": 3, "wood": 3, "fiber": 2, "stone": 2}


class Folk:
    def __init__(self, sim, player, seed: int, name="Neighbour"):
        self.sim, self.p = sim, player
        self.rng = random.Random(seed * 17 + 3)
        self.a = sim.login(name, f"folk-{name}")
        self.a.online = False
        self.next = sim.tick + self.rng.randint(*EVERY)
        self.until = self.act = 0
        self.owed = False                       # given something by the player: it may give back

    @property
    def on(self) -> bool:
        return self.a.online

    def dist(self) -> float:
        return math.dist((self.a.x, self.a.y), (self.p.x, self.p.y))

    def tick(self):
        sim, a = self.sim, self.a
        if not a.online:
            if sim.tick >= self.next and self.p.hp > 0:
                self.arrive()
            return
        if sim.tick >= self.until or self.dist() > 30:
            a.online = False
            self.next = sim.tick + self.rng.randint(*EVERY)
            return
        for e in a.events:
            if e.get("type") == "give" and e.get("to") == a.name:
                self.owed = True
        a.events = []
        if sim.tick < self.act or a.action is not None:
            return
        self.act = sim.tick + self.rng.randint(*ACT)
        d, r = self.dist(), self.rng.random()
        if self.owed and d <= 2 and r < 0.6:
            self.owed = False
            return self.give()
        if d > 6 or r < 0.25:
            ang = self.rng.uniform(0, 2 * math.pi)
            return sim.submit(a, "move_to", {"x": self.p.x + round(3 * math.cos(ang)), "y": self.p.y + round(3 * math.sin(ang))})
        if r < 0.35:
            return sim.submit(a, "say", {"text": self.rng.choice(("hey", "hi there", "morning", "yo"))})
        if r < 0.45:
            return sim.submit(a, "say", {"text": f"got any {self.rng.choice(ASKS)} to spare?"})
        if r < 0.55 and d <= 2:
            return self.give()
        for name, (dx, dy) in DIRS.items():               # work: whatever is beside it
            if sim.world.block(a.x + dx, a.y + dy) in ("tree", "rock", "thicket", "berry_bush"):
                return sim.submit(a, "dig", {"dir": name})
        sim.submit(a, "move_to", {"x": a.x + self.rng.randint(-4, 4), "y": a.y + self.rng.randint(-4, 4)})

    def give(self):
        item = next((k for k in GIFTS if self.a.count(k)), None)
        if item:
            self.sim.submit(self.a, "give", {"target": self.p.id, "item": item, "n": 1})

    def arrive(self):
        sim, a, p = self.sim, self.a, self.p
        for _ in range(30):
            ang = self.rng.uniform(0, 2 * math.pi)
            x, y = p.x + round(8 * math.cos(ang)), p.y + round(8 * math.sin(ang))
            if sim.world.passable(x, y):
                break
        else:
            self.next = sim.tick + sec(30)
            return
        a.inventory.clear()
        a.inventory.update(GIFTS)
        a.x, a.y, a.z, a.hp, a.online = x, y, 99, 20, True
        a.events = []
        self.until, self.act, self.owed = sim.tick + self.rng.randint(*STAY), sim.tick + sec(2), False
