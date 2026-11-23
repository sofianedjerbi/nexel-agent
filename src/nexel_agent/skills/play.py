"""A skill in play: `skill NAME [ARGS]` in a plan runs the library's skill (model-written) to its end, its first
error, or a card firing; the body is busy meanwhile, so no second waits on the model. Its own done_when decides win
or loss, which the library keeps. Mixed into Thinker; the skill's code runs in a thread, each of its commands goes
to the game (or a verb) on the player's own loop."""

import asyncio

from ..goals import missing
from ..items import by_id
from ..plans import LOCAL_SPECS, parse_command
from ..verbs import VERBS, locate
from .library import Library
from .sandbox import Sandbox

MAX_S = 180.0


class Skills:
    def skills_init(self, data_dir):
        self.lib = Library(data_dir / "_skills")
        self.skill_stop = False

    def skill_lines(self) -> str:
        return self.lib.lines(6) if self.lib.index else ""

    async def run_skill(self, game, item) -> dict:
        w = str(item.args.get("text", "")).split()
        name = w[0] if w else ""
        src = self.lib.sources()
        if name not in src:
            return {"ok": False, "error": "no_skill", "detail": f"no skill {name!r}; skills: {', '.join(self.lib.offered()) or 'none'}"}
        args = tuple(int(x) if x.lstrip("-").isdigit() else x for x in w[1:])
        loop, self.skill_stop = asyncio.get_running_loop(), False

        def do(cmd):
            return asyncio.run_coroutine_threadsafe(self.skill_command(game, cmd), loop).result(timeout=120)

        def observe():
            return asyncio.run_coroutine_threadsafe(self.observe(game), loop).result(timeout=30)
        box = Sandbox(do, observe, max_actions=300, max_s=MAX_S, stop=lambda: self.skill_stop)
        run = await asyncio.to_thread(box.run, name, src[name], args, src)
        obs = await self.observe(game)
        test = self.lib.index[name].get("done_when") or {}
        won = (any(max(abs(p[0] - obs["self"]["x"]), abs(p[1] - obs["self"]["y"])) <= 3 for p in locate(obs, test["placed"]))
               if "placed" in test else not missing(test, obs))
        if not (run["error"] or "").startswith("stopped"):
            self.lib.outcome(name, won)
        last = " | ".join(run["trace"][-3:])
        if won:
            return {"ok": True, "result": {"ticks": 0, "changed": {}, "text": f"skill {name} done ({len(run['trace'])} actions)"}}
        return {"ok": False, "error": "skill_failed", "detail": f"{run['error'] or 'returned ' + repr(run['value'])}; last: {last}"[:200]}

    async def skill_command(self, game, cmd) -> dict:
        c = parse_command(by_id(cmd, self.obs))
        if c is None:
            return {"ok": False, "error": "bad_command", "detail": f"{cmd!r} is not a command"}
        tool, args = c
        if tool in LOCAL_SPECS:
            return {"ok": False, "error": "not_in_a_skill", "detail": f"{tool} is not for skills"}
        if tool in VERBS:
            return await self.verb(game, tool, args)
        return await game.act(tool, **args)
