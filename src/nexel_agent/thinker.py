"""LLM thinker harness, real time: the model plans a few actions per call, the body never waits for it.

Two loops share one queue. The PLANNER asks the model for a short plan ({"goal", "do": [commands]}) on the
newest observation, streams it, and queues each command the moment its string closes; it starts the next
call as soon as the queue runs low, telling the model what is already queued (pipelining). The BODY runs the
queue back to back, re-checking every result: a failure, a hit, a reflex drops the rest of that plan (and a
plan still streaming from before it). When the queue is empty the body does the goal's obvious next step
(filler.py: walk, gather, step away), cut short the moment a plan arrives. Someone talking nearby gets a
tiny separate model call and the answer goes to the front of the queue.

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

from .aim import facing, fail_key, gone, just_failed, note_outcome, reflex
from .client import ApiError, Client
from .fallback import climbing, fallback, stuck
from .fastprompt import stable_prefix
from .lessons import Lessons
from .planner import Item, Planner
from .planwatch import PlanWatch
from .plans import command_text, parse_plan  # noqa: F401  (parse_plan: tools, tests)
from .playstats import summary
from .policy import feedback
from .watchdog import Watchdog, stall_line

log = logging.getLogger("thinker")
MODEL_URL = "http://127.0.0.1:8080/v1/chat/completions"
ANCHOR_S = 8.0                                    # after a planned walk, the filler stays near its end this long
MIN_GAP_S = 0.33                                  # between two actions sent, whatever their source (~3 a second)


class Thinker(Planner):
    def __init__(self, name, persona, token=None, game_url="ws://127.0.0.1:7777/agent", model_url=MODEL_URL,
                 model="default_model", data_dir=Path("agents"), temperature=0.3, max_tokens=120, refill_at=1,
                 chat=True, decision_log=None):
        self.name, self.persona = name, persona
        self.token = token or f"thinker-{name}"
        self.game_url, self.model_url, self.model = game_url, model_url, model
        self.dir = Path(data_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.notepad_path = self.dir / "notepad.md"
        self.lessons = Lessons(self.dir / "lessons.json")
        self.temperature, self.max_tokens, self.refill_at, self.chat = temperature, max_tokens, refill_at, chat
        self.decision_log = open(decision_log or self.dir / "decisions.jsonl", "a")
        self.prefix = stable_prefix(name, persona)          # byte-identical every call: the server's prompt cache keeps it
        self.rng = random.Random(hash(name) & 0xFFFF)
        self.memory: dict = {}
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
        self.last_hp = None
        self.planning = False
        self.last_goto = None                                 # the last planned move_to that worked
        self.failed: dict[tuple, list] = {}                   # fail_key -> [times, error]; forgotten when the pack gains a kind
        self.kinds: set[str] = set()
        self.watch = PlanWatch()                              # plans asked again that changed nothing
        self.current: Item | None = None                      # the planned command being played now
        self.last_skip = ""                                   # why the last planned command was not played
        self.anchor: tuple | None = None                      # (x, y, until): where the plan just walked; the filler stays near
        self.dog = Watchdog()                                 # no progress for 20 s, whatever the source: that routine is banned here
        self.stats = Counter()
        self.calls: list[dict] = []
        self.busy: list[tuple[float, float, str]] = []      # (start, end, source) of every action on the server

    # ------------------------------------------------------------ notepad
    def notepad(self) -> str:
        return self.notepad_path.read_text() if self.notepad_path.exists() else ""

    def add_note(self, line: str):
        line = line.strip().replace("\n", " ")[:200]
        lines = [x for x in self.notepad().splitlines() if x.strip()]
        if line and line not in lines:
            self.notepad_path.write_text("\n".join((lines + [line])[-30:]) + "\n")

    # --------------------------------------------------------------- body
    def abort(self, why: str):
        if any(i.source in ("model", "align") for i in self.queue) or self.planning:
            self.stats["aborts"] += 1
            self.history.append(f"(plan stopped: {why})")
        self.epoch += 1
        self.queue = deque(i for i in self.queue if i.source == "chat")

    async def observe(self, game):
        obs = await game.observe()
        me = obs["self"]
        for e in obs["events"]:
            if e.get("type") in ("say", "shout") and e.get("from") != self.name:
                self.heard.append(f"{e['from']}: \"{str(e.get('text', ''))[:80]}\"")
                if self.chat:
                    self.on_chat(e["from"], str(e.get("text", ""))[:200], (me["x"] + e.get("dx", 0), me["y"] + e.get("dy", 0)))
            elif e.get("type") == "attack" and e.get("target") == self.name:
                self.on_hit(e, me)
        if self.last_hp is not None and self.last_hp - me["hp"] >= 3:
            self.abort(f"you lost {self.last_hp - me['hp']} health")
        self.last_hp = me["hp"]
        kinds = set(me["inventory"])
        if kinds - self.kinds:                       # a new thing in the pack (a pickaxe...): old failures may work now
            self.failed.clear()
        self.kinds = kinds
        self.obs = obs
        return obs

    urgent = staticmethod(reflex)

    def next_item(self, obs) -> Item:
        u = self.urgent(obs)
        if u:
            if u[0] != "eat" or obs["self"]["hunger"] <= 3:   # eating while hungry: the plan goes on after it
                self.abort(u[2])
            return Item(*u[:2], "reflex", u[2])
        me = obs["self"]
        if it := climbing(self, obs):
            return it
        while self.queue:
            it = self.queue.popleft()
            if it.epoch != self.epoch and it.source != "chat":
                continue
            if it.at and (me["x"], me["y"]) != tuple(it.at):
                if abs(it.at[0] - me["x"]) + abs(it.at[1] - me["y"]) > 12 or stuck(obs, self.memory):
                    self.abort(f"'{command_text(it.tool, it.args)}' was meant from {it.at}, no going back there")
                    break
                self.queue.appendleft(Item(it.tool, it.args, it.source, it.why, None, it.epoch))
                return Item("move_to", {"x": it.at[0], "y": it.at[1]}, "align", f"back to {it.at} for the plan",
                            None, it.epoch)
            if it.tool == "dig" and self.last_goto and self.gone(obs, it):
                d = facing(obs, self.last_goto)
                if d:                                # "move_to <the tree>, dig": it stopped beside it, so dig the tree
                    it = Item(it.tool, {**it.args, "dir": d}, it.source, it.why, None, it.epoch)
                    self.stats["dig_turned"] += 1
            if self.gone(obs, it):
                self.stats["skipped"] += 1
                self.last_skip = f"'{command_text(it.tool, it.args)}': nothing to dig there"
                self.tell(f"(skipped {self.last_skip})")
                continue
            known = self.failed.get(fail_key(obs, it))
            again = just_failed(self.memory, it.tool, it.args, time.monotonic())
            if (known and known[0] >= 2) or again:   # failed here twice already, or a moment ago: not again now
                self.stats["known_fail"] += 1
                self.last_skip = f"'{command_text(it.tool, it.args)}' failed " + (
                    f"{known[0]}x here before: {known[1]}" if known and known[0] >= 2 else "a moment ago")
                self.abort(self.last_skip)
                break
            return it
        return fallback(self, obs)

    gone = staticmethod(gone)

    async def body_loop(self, game):
        last, routine = 0.0, ""
        while True:
            try:
                obs = await self.observe(game)
            except ApiError as e:
                if e.code != "rate_limited":
                    raise
                await asyncio.sleep(0.5)
                continue
            me, now = obs["self"], time.monotonic()
            if routine and (stall := self.dog.record(now, me["x"], me["y"], me["inventory"], routine)):
                self.on_stall(stall)
            self.new_plan.clear()
            item = self.next_item(obs)
            routine = item.routine or item.source
            await asyncio.sleep(max(0.0, last + MIN_GAP_S - time.monotonic()))   # at most ~3 actions a second
            last = time.monotonic()
            await self.execute(game, item, obs)

    def on_stall(self, stall):
        """A routine went nowhere for WINDOW_S: banned here (watchdog), its state dropped, the model told."""
        self.stats["stalls"] += 1
        for k in ("escape_target", "stair", "climb_to", "stair_dir", "swim_heading", "climb_steps"):
            self.memory.pop(k, None)
        self.memory["no_path"] = 0
        if stall["routine"] in ("model", "align"):
            self.abort("it went nowhere")
        self.tell(stall_line(stall))
        self.decision_log.write(json.dumps({"kind": "stall", **stall}) + "\n")
        log.info("%s [stall] %s", self.name, stall_line(stall))

    async def execute(self, game, item: Item, obs):
        t0 = time.monotonic()
        self.current = item if item.source == "model" else None
        task = asyncio.create_task(game.act(item.tool, **item.args))
        if item.source == "filler":                      # cut short by a plan (the next act replaces it on the server)
            waiter = asyncio.create_task(self.new_plan.wait())
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if task not in done:
                task.add_done_callback(lambda t: t.cancelled() or t.exception())
                self.busy.append((t0, time.monotonic(), "filler"))
                self.stats["filler:cut"] += 1
                self._log(item, obs, {"ok": True, "result": {"ticks": 0}}, "(cut short by a plan)", t0)
                return
        try:
            reply = await task
        except asyncio.CancelledError:
            task.cancel()
            raise
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
            self.last_goto = (item.args["x"], item.args["y"]) if item.tool == "move_to" and ok else None
            if self.last_goto:
                me = reply["result"].get("changed", {}).get("pos") or self.last_goto
                self.anchor = (me[0], me[1], t1 + ANCHOR_S)
        if item.source != "filler" or not ok:
            self.history.append(line[:260])
        note_outcome(self.memory, self.failed, obs, item, reply, t1)
        if not ok and item.source in ("model", "align") and reply.get("error") != "interrupted":
            self.abort(f"{command_text(item.tool, item.args)} failed: {reply.get('error')}")
        self._log(item, obs, reply, line, t0)
        if not ok and t1 - t0 < 0.15:
            await asyncio.sleep(0.25)                    # a refusal takes no time: do not hammer the server

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
