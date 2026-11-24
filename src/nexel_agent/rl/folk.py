"""Peaceful neighbours in the sim (part of the environment, as the rival is: scripted, never the policy). Now and
then one comes by and spends a few minutes around the player (now and then sticking right beside it, as a person
who follows you about): wanders near, works (chops, digs), and once in a
while says hello, asks for something, proposes to team up, or hands something over; given something, it sometimes
gives back; asked for something it carries, it brings it over or says no. It carries a small pack, as a player
does, and drops it when killed. Hit by the player, it hits back a few seconds, then goes; and it remembers, as do
the neighbours who saw it: they say no to the player's asks, give nothing, keep their distance, and now and then
come back to set upon it - for a few minutes: a grudge fades, faster with a gift, at once when the player helps it.
They say plain short lines; on the live server real players and the LLM's chat take their place."""

import math
import random

from nexel_worlds.sim.constants import sec
from nexel_worlds.tiles import DIRS
from .facts import Attack, Give, Say, events, named

EVERY = (sec(60), sec(200))          # between two visits
STAY = (sec(120), sec(300))
ACT = (sec(3), sec(8))               # between two things it does
ASKS = ("wood", "stone", "berries", "fiber", "plank", "dirt")
PACK = ("berries", "wood", "fiber", "stone", "seed", "dirt")   # what a neighbour may carry, a few of each at most
ANGRY = sec(8)                       # (a blow is seen by whoever is in view of it: the game's own events)
STICKY = 0.4                         # the share of visits it sticks right beside the player (a person who follows you about)
GRUDGE = sec(240)                    # a grudge fades in a few minutes,
GIFT_EASES = sec(120)                # faster when the player gives it something; at once when the player helps it in a fight


class Folk:
    def __init__(self, sim, player, seed: int, name="Neighbour", town=None):
        """town: the list of all the neighbours (who saw what)."""
        self.sim, self.p = sim, player
        self.rng = random.Random(seed * 17 + 3)
        self.a = sim.login(name, f"folk-{name}")
        self.a.online = False
        self.next = sim.tick + self.rng.randint(*EVERY)
        self.until = self.act = 0
        self.owed = False                       # given something by the player: it may give back
        self.asked, self.proposed = None, False  # asked by the player for something it said yes to; asked to team up
        self.angry = 0                          # hit by the player: till when it hits back
        self.grudge_until = 0                   # it was hit by the player, or saw a neighbour hit: till when it minds
        self.set_upon = False                   # the rival is going for it now (the env says)
        self.sticky = False                     # this visit, it sticks right beside the player
        self.shadowing = False                  # this episode, it is always beside the player (it comes back after leaving)
        self.town = town if town is not None else [self]

    @property
    def grudge(self) -> bool:
        return self.sim.tick < self.grudge_until

    @grudge.setter
    def grudge(self, held: bool):
        self.grudge_until = self.sim.tick + GRUDGE if held else 0

    @property
    def on(self) -> bool:
        return self.a.online

    def dist(self) -> float:
        return math.dist((self.a.x, self.a.y), (self.p.x, self.p.y))

    def go(self, wait=EVERY):
        self.a.online, self.angry = False, 0
        self.next = self.sim.tick + (sec(60) if self.shadowing else self.rng.randint(*wait))

    def tick(self):
        sim, a = self.sim, self.a
        if not a.online:
            if sim.tick >= self.next and self.p.hp > 0:
                self.arrive()
            return
        if a.hp <= 0 or sim.tick >= self.until or self.dist() > 30:
            return self.go()
        for ev in events(a.events):
            match ev:
                case Give(to=to) if to == a.name:
                    self.owed = True
                    self.grudge_until -= GIFT_EASES                           # (a gift softens it)
                case Attack(by=by, target="Rival") if by == self.p.name and self.set_upon:
                    self.grudge = False                                       # (it stood by it against the rival: forgiven)
                case Attack(by=by, target=tgt) if by == self.p.name and tgt in {f.a.name for f in self.town}:
                    if tgt == a.name:
                        self.angry, self.asked = sim.tick + ANGRY, None      # (hit by the player: it hits back, then goes)
                    self.grudge = True                                        # (hit, or saw it: within earshot of the blow)
                case Say(src=src, shout=False) if src == self.p.name:      # asked for something?
                    it = next(iter(named(ev.text, a.inventory)), None)
                    if it and not self.grudge and self.rng.random() < 0.6:
                        self.asked = it
                    elif it or "spare" in ev.text:
                        sim.submit(a, "say", {"text": self.rng.choice(("sorry, need it", "no, can't", "not now"))})
        a.events = []
        if self.angry:
            if sim.tick >= self.angry:
                return self.go((EVERY[0] * 2, EVERY[1] * 2))
            if sim.tick >= self.act:
                self.act = sim.tick + sec(1)
                if self.dist() <= 2.5:
                    sim.submit(a, "attack", {"target": self.p.id})
                else:
                    sim.submit(a, "move_to", {"x": self.p.x, "y": self.p.y})
            return
        if sim.tick < self.act or a.action is not None:
            return
        self.act = sim.tick + self.rng.randint(*ACT)
        d, r = self.dist(), self.rng.random()
        if self.sticky and not self.grudge and d > 1.5 and r < 0.7:     # (back to its place beside the player)
            self.act = sim.tick + sec(1)
            return sim.submit(a, "move_to", {"x": self.p.x + 1, "y": self.p.y})
        if self.grudge:                                  # (keeps its distance, works on its own)
            if d < 8:
                ang = math.atan2(a.y - self.p.y, a.x - self.p.x)
                return sim.submit(a, "move_to", {"x": self.p.x + round(10 * math.cos(ang)), "y": self.p.y + round(10 * math.sin(ang))})
            return self.work()
        if self.asked:                                   # (said yes: it brings it over)
            if d <= 2:
                item, self.asked = self.asked, None
                return sim.submit(a, "give", {"target": self.p.id, "item": item, "n": 1})
            return sim.submit(a, "move_to", {"x": self.p.x + 1, "y": self.p.y})
        if not self.proposed and r < 0.05:
            self.proposed = True
            return sim.submit(a, "say", {"text": self.rng.choice(("want to team up?", "we could work together", "stick with me?"))})
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
        self.work()

    def work(self):
        sim, a = self.sim, self.a
        for name, (dx, dy) in DIRS.items():               # whatever is beside it
            if sim.world.block(a.x + dx, a.y + dy) in ("tree", "rock", "thicket", "berry_bush"):
                return sim.submit(a, "dig", {"dir": name})
        sim.submit(a, "move_to", {"x": a.x + self.rng.randint(-4, 4), "y": a.y + self.rng.randint(-4, 4)})

    def give(self):
        item = next((k for k in PACK if self.a.count(k)), None)
        if item:
            self.sim.submit(self.a, "give", {"target": self.p.id, "item": item, "n": 1})

    def shadow(self):
        """This episode it stays right beside the player all along (a person who follows you about), from the start."""
        self.shadowing, self.next = True, self.sim.tick + sec(5)

    def arrive(self):
        sim, a, p = self.sim, self.a, self.p
        for _ in range(30):
            ang = self.rng.uniform(0, 2 * math.pi)
            x, y = p.x + round(8 * math.cos(ang)), p.y + round(8 * math.sin(ang))
            if sim.stand(x, y):
                break
        else:
            self.next = sim.tick + sec(30)
            return
        a.inventory.clear()
        for k in self.rng.sample(PACK, 3):                 # (a small pack, as a player carries)
            a.inventory[k] = self.rng.randint(1, 3)
        a.x, a.y, a.z, a.hp, a.online = x, y, 99, 20, True
        a.events = []
        self.until, self.act, self.owed = sim.tick + self.rng.randint(*STAY), sim.tick + sec(2), False
        self.asked, self.proposed = None, False
        self.sticky = self.shadowing or self.rng.random() < STICKY
        if self.shadowing:
            self.until = sim.tick + sec(24 * 3600)
        self.angry = sim.tick + ANGRY if self.grudge and self.rng.random() < 0.3 else 0     # (back to settle it)
