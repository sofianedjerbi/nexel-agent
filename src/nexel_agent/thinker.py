"""LLM thinker harness, real time: the model plans a few actions per call and decides every one of them.

Two loops share one queue. The PLANNER asks the model for a short plan ({"goal", "do": [commands]}) on the
newest observation, streams it, and queues each command the moment its string closes; it starts the next
call as soon as the queue runs low, telling the model what is already queued (pipelining). The BODY plays the
queue back to back and reports every result truthfully; a failure or a hit drops the rest of that plan, and the
next plan is made knowing it. With nothing queued the body waits for the model: no scripted filler, reflex or
escape acts in its place. Someone talking nearby gets a small separate model call (chat.py). The player's memory
(memory.py) is its own: it writes notes with `remember`, drops them with `forget`, and summarises its days.

    python -m megastructure.thinker --name Vex --persona "scavenger who trades" --model-url http://127.0.0.1:8080/v1/chat/completions
"""

import argparse
import asyncio
import json
import logging
import random
import time
from collections import Counter, deque
from pathlib import Path

import httpx

from .aim import just_failed, note_outcome
from .client import ApiError, Client
from .fastprompt import stable_prefix
from .lessons import Lessons
from .memory import Memory
from .planner import LOCAL, Item, Planner
from .planwatch import PlanWatch
from .plans import command_text, parse_plan  # noqa: F401  (parse_plan: tools, tests)
from .playstats import summary
from .policy import feedback
from .senses import Body
from .watchdog import Watchdog, stall_line

log = logging.getLogger("thinker")
MODEL_URL = "http://127.0.0.1:8080/v1/chat/completions"
MIN_GAP_S = 0.33                                  # between two actions sent (~3 a second)
IDLE_WAIT_S = 0.5                                 # nothing queued: look again this often while the model thinks


class Thinker(Planner):
    def __init__(self, name, persona, token=None, game_url="ws://127.0.0.1:7777/agent", model_url=MODEL_URL,
                 model="default_model", data_dir=Path("agents"), temperature=0.3, max_tokens=120, refill_at=1,
                 chat=True, decision_log=None):
        self.name, self.persona = name, persona
        self.token = token or f"thinker-{name}"
        self.game_url, self.model_url, self.model = game_url, model_url, model
        self.dir = Path(data_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lessons = Lessons(self.dir / "lessons.json")
        self.mem = Memory(self.dir / "memory.json")
        self.body = Body()
        self.temperature, self.max_tokens, self.refill_at, self.chat = temperature, max_tokens, refill_at, chat
        self.decision_log = open(decision_log or self.dir / "decisions.jsonl", "a")
        self.prefix = stable_prefix(name, persona)          # byte-identical every call: the server's prompt cache keeps it
        self.rng = random.Random(hash(name) & 0xFFFF)
        self.memory: dict = {}                                # the body's plumbing state (last failure, no_path count)
        self.queue: deque[Item] = deque()
        self.epoch = 0
        self.goal: str | None = None
        self.obs: dict | None = None
        self.heard: deque[str] = deque(maxlen=4)
        self.history: deque[str] = deque(maxlen=4)
        self.recent: deque[str] = deque(maxlen=20)
        self.new_plan = asyncio.Event()
        self.chat_task: asyncio.Task | None = None
        self.chat_next: tuple | None = None
        self.planning = False
        self.watch = PlanWatch()                              # plans asked again that changed nothing: told back
        self.current: Item | None = None                      # the planned command being played now
        self.last_skip = ""
        self.dog = Watchdog()                                 # no progress for 20 s: the model is told, nothing more
        self.stats = Counter()
        self.calls: list[dict] = []
        self.busy: list[tuple[float, float, str]] = []      # (start, end, source) of every action on the server

    # ------------------------------------------------------------ notes (the model's "note" field)
    def add_note(self, line: str):
        me = (self.obs or {}).get("self", {})
        self.mem.remember(line.strip().replace("\n", " "), (me.get("x"), me.get("y")) if me else None,
                          str((self.obs or {}).get("time", "")))

    # --------------------------------------------------------------- body
    def abort(self, why: str):
        if any(i.source == "model" for i in self.queue) or self.planning:
            self.stats["aborts"] += 1
            self.history.append(f"(plan stopped: {why})")
        self.epoch += 1
        self.queue = deque(i for i in self.queue if i.source == "chat")

    async def observe(self, game):
        obs = await game.observe()
        me = obs["self"]
        here, when = (me["x"], me["y"]), str(obs.get("time", ""))
        self.body.feel(obs)
        for e in obs["events"]:
            at = (me["x"] + e.get("dx", 0), me["y"] + e.get("dy", 0))
            if e.get("type") in ("say", "shout") and e.get("from") != self.name:
                text = str(e.get("text", ""))[:200]
                self.heard.append(f"{e['from']}: \"{text[:80]}\"")
                self.mem.heard(e["from"], text, at, when)
                if self.chat:
                    self.on_chat(e["from"], text, at)
            elif e.get("type") == "attack" and e.get("target") == self.name:
                self.mem.hit_by(str(e.get("attacker")), e.get("damage"), e.get("weapon"), here, when, me["hp"])
                self.on_hit(e, me)
            elif e.get("type") == "death" and e.get("name") == self.name:
                self.mem.add("death", f"you died: {e.get('cause')}" + (f" by {e['by']}" if e.get("by") else "")
                             + ", your pack dropped there", e.get("by"), at, when)
                self.abort("you died")
            elif e.get("type") == "give" and e.get("to") == self.name:
                g = self.mem.person(str(e.get("from")))["gave_you"]
                g[str(e.get("item"))] = g.get(str(e.get("item")), 0) + int(e.get("n") or 1)
                self.mem.add("gift", f"{e.get('from')} gave you {e.get('n')} {e.get('item')}", e.get("from"), at, when)
        self.obs = obs
        return obs

    def next_item(self, obs) -> Item | None:
        """The next queued command, or None (the model has not planned one yet)."""
        me = obs["self"]
        while self.queue:
            it = self.queue.popleft()
            if it.epoch != self.epoch and it.source != "chat":
                continue
            if it.at and (me["x"], me["y"]) != tuple(it.at):     # (a relative command from somewhere else would aim wrong)
                self.tell(f"('{command_text(it.tool, it.args)}' was planned from {tuple(it.at)}, you are at "
                          f"({me['x']},{me['y']}): not done)")
                self.stats["misplaced"] += 1
                continue
            if just_failed(self.memory, it.tool, it.args, time.monotonic()):
                self.stats["backoff"] += 1                        # the same command failed a moment ago: not resent at once
                self.abort(f"'{command_text(it.tool, it.args)}' failed a moment ago")
                return None
            return it
        return None

    async def body_loop(self, game):
        last = 0.0
        while True:
            try:
                obs = await self.observe(game)
            except ApiError as e:
                if e.code != "rate_limited":
                    raise
                await asyncio.sleep(0.5)
                continue
            me, now = obs["self"], time.monotonic()
            if stall := self.dog.record(now, me["x"], me["y"], me["inventory"], "you"):
                self.on_stall(stall)
            self.new_plan.clear()
            item = self.next_item(obs)
            if item is None:                             # the model has not said what to do yet: wait for it
                try:
                    await asyncio.wait_for(self.new_plan.wait(), IDLE_WAIT_S)
                except asyncio.TimeoutError:
                    pass
                continue
            await asyncio.sleep(max(0.0, last + MIN_GAP_S - time.monotonic()))   # at most ~3 actions a second
            last = time.monotonic()
            await self.execute(game, item, obs)

    def on_stall(self, stall):
        """Nothing moved or changed for WINDOW_S: the model is told (what to do about it is its own choice)."""
        self.stats["stalls"] += 1
        self.memory["no_path"] = 0
        self.tell(stall_line(stall))
        self.decision_log.write(json.dumps({"kind": "stall", **stall}) + "\n")
        log.info("%s [stall] %s", self.name, stall_line(stall))

    async def execute(self, game, item: Item, obs):
        t0 = time.monotonic()
        self.current = item if item.source == "model" else None
        if item.tool in LOCAL:
            reply = self.local(item, obs)
        else:
            reply = await game.act(item.tool, **item.args)
        t1 = time.monotonic()
        self.current = None
        self.busy.append((t0, t1, item.source))
        ok = bool(reply.get("ok"))
        feedback(self.memory, reply, obs["tick"])
        self.lessons.learn(item.tool, item.args, reply)
        line = self.result_line(item, reply, obs)
        self.stats[f"{item.source}:{'ok' if ok else 'fail'}"] += 1
        if item.source == "model":
            self.recent.append(command_text(item.tool, item.args))
        self.history.append(line[:260])
        note_outcome(self.memory, obs, item, reply, t1)
        if not ok and reply.get("error") != "interrupted":
            self.abort(f"{command_text(item.tool, item.args)} failed: {reply.get('error')}")
        self._log(item, obs, reply, line, t0)
        if not ok and t1 - t0 < 0.15:
            await asyncio.sleep(0.25)                    # a refusal takes no time: do not hammer the server

    def local(self, item, obs) -> dict:
        """remember / forget: the model's own memory, kept by the harness (no game action)."""
        me = obs["self"]
        if item.tool == "remember":
            n = self.mem.remember(str(item.args.get("text", "")), (me["x"], me["y"]), str(obs.get("time", "")))
            return {"ok": True, "result": {"ticks": 0, "changed": {}, "text": f"noted [{n}]"}}
        n = item.args.get("n")
        if isinstance(n, int) and self.mem.forget(n):
            return {"ok": True, "result": {"ticks": 0, "changed": {}, "text": f"forgot [{n}]"}}
        return {"ok": False, "error": "no_note", "detail": f"no note [{n}]; your notes: {self.mem.notes_text() or 'none'}"}

    def _log(self, item, obs, reply, line, t0):
        me = obs["self"]
        rec = {"t": time.time(), "dur_s": round(time.monotonic() - t0, 2), "tick": obs["tick"], "agent": self.name,
               "source": item.source, "pos": [me["x"], me["y"]], "hp": me["hp"], "hunger": me["hunger"],
               "goal": self.goal, "thought": item.why, "tool": item.tool, "args": item.args, "ok": bool(reply.get("ok")),
               "error": reply.get("error"), "result": line}
        self.decision_log.write(json.dumps(rec) + "\n")
        self.decision_log.flush()
        log.info("%s [%s %.1fs] %s | %s", self.name, item.source, rec["dur_s"], item.why[:100], line[:140])

    # --------------------------------------------------------------- run
    async def run(self, minutes: float | None = None):
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=4)) as http, Client(self.game_url) as game:
            await game.hello(self.name, self.token, self.persona[:200])
            log.info("%s joined", self.name)
            return await self.play(game, http, minutes)

    async def play(self, game, http, minutes: float | None = None) -> dict:
        """Run the body and the planner until `minutes` are up (or one of them fails); returns the summary."""
        self.http = http
        t0 = time.monotonic()
        tasks = [asyncio.create_task(self.body_loop(game)), asyncio.create_task(self.plan_loop(http))]
        try:
            done, _ = await asyncio.wait(tasks, timeout=minutes * 60 if minutes else None,
                                         return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                t.result()
        finally:
            for t in tasks + ([self.chat_task] if self.chat_task else []):
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            s = summary(self.busy, self.calls, time.monotonic() - t0, self.stats)
            (self.dir / "summary.json").write_text(json.dumps(s, indent=1))
            log.info("%s summary %s", self.name, json.dumps(s))
            self.decision_log.flush()
        return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--persona", default="a curious scavenger who wants to survive and build a home")
    p.add_argument("--game", default="ws://127.0.0.1:7777/agent")
    p.add_argument("--model-url", default=MODEL_URL)
    p.add_argument("--model", default="default_model")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--data", default="agents")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--no-chat", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(Thinker(a.name, a.persona, game_url=a.game, model_url=a.model_url, model=a.model, data_dir=Path(a.data),
                        temperature=a.temperature, max_tokens=a.max_tokens, chat=not a.no_chat).run(a.minutes))


if __name__ == "__main__":
    main()
