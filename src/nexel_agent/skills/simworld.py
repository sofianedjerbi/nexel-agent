"""A body for skills in the in-process simulation (the forge's test bench): a fresh generated world per seed, the
player at its spawn in the morning, with our verbs (verbs.py) and the game's tools, at full speed (no wall clock)."""

import asyncio
import math
import tempfile
from pathlib import Path

from ..items import by_id
from ..plans import LOCAL_SPECS, parse_command
from megastructure.sim import Simulation
from megastructure.sim.constants import DAY_TICKS
from ..fixes import look_far_line, look_far_things
from ..verbs import VERBS, Verbs, note_seen


class SimGame:
    """The sim behind the client's async observe/act, shaped as the server replies."""

    def __init__(self, sim, a):
        self.sim, self.a = sim, a

    async def observe(self):
        return self.sim.observe(self.a)

    async def act(self, tool, **args):
        act = self.sim.submit(self.a, tool, dict(args))
        for _ in range(3000):
            if act.result is not None:
                r = dict(act.result)
                if r.pop("ok", True):
                    return {"ok": True, "result": r}
                return {"ok": False, "error": r.get("error"), "detail": r.get("detail"), "result": r}
            self.sim.step()
        return {"ok": False, "error": "timeout", "detail": "the action did not end in 300 s of game time"}


class Host(Verbs):
    """What the verbs need from a player: a look, and a map memory."""

    def __init__(self):
        self.memory = {"seen": {}}

    async def observe(self, game):
        obs = await game.observe()
        note_seen(obs, self.memory["seen"])
        return obs


class SimBody:
    def __init__(self, seed: int, dir_: Path | None = None, inventory=None, max_ticks=6000):
        self.dir = Path(dir_ or tempfile.mkdtemp(prefix="forge"))
        self.sim = Simulation(self.dir / f"w{seed}.sqlite", seed=seed)
        self.sim.keep_loaded = False
        self.a = self.sim.login("Forge", "forge-token")
        self.sim.tick = self.sim.tick // DAY_TICKS * DAY_TICKS + int(0.3 * DAY_TICKS)   # (morning)
        self.a.inventory.update(inventory or {})
        self.game, self.host = SimGame(self.sim, self.a), Host()
        self.loop = asyncio.new_event_loop()
        self.t0, self.max_ticks = self.sim.tick, max_ticks

    def observe(self):
        return self.sim.observe(self.a)

    def do(self, cmd: str) -> dict:
        if self.sim.tick - self.t0 > self.max_ticks:
            return {"ok": False, "error": "out_of_time", "detail": f"{self.max_ticks // 600} min of game time used"}
        c = parse_command(by_id(cmd, self.observe()))
        if c is None:
            return {"ok": False, "error": "bad_command", "detail": f"{cmd!r} is not a command"}
        tool, args = c
        if tool in LOCAL_SPECS:
            return {"ok": True, "result": {"ticks": 0, "changed": {}}}
        if tool in VERBS:
            return self.loop.run_until_complete(self.host.verb(self.game, tool, args))
        r = self.loop.run_until_complete(self.game.act(tool, **args))
        if tool == "look_far" and r.get("ok"):            # (what it saw goes to the map memory, as in play)
            me = self.observe()["self"]
            r["result"]["text"] = look_far_line(args, r["result"], me, self.host.memory["seen"])
            r["result"]["things"] = look_far_things(args, r["result"], me)
        else:
            note_seen(self.observe(), self.host.memory["seen"])
        return r

    def placed_near(self, block, r=3) -> bool:
        return any(self.sim.world.block(self.a.x + dx, self.a.y + dy) == block for dx in range(-r, r + 1) for dy in range(-r, r + 1))

    def resources(self, r=40) -> dict:
        """How many trees and rocks lie within r tiles of the spawn (to pick test worlds with something to work on)."""
        out = {"tree": 0, "rock": 0}
        for y in range(self.a.y - r, self.a.y + r + 1, 2):
            for x in range(self.a.x - r, self.a.x + r + 1, 2):
                b = self.sim.world.block(x, y)
                if b in out and math.dist((x, y), (self.a.x, self.a.y)) <= r:
                    out[b] += 1
        return out

    def close(self):
        self.loop.close()
