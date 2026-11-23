"""The thinker's model side: plan calls streamed into the queue (pipelined on the newest observation). Mixed into
Thinker (thinker.py), whose body loop runs the queue; the chat reply and the diary are in chat.py.
"""

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass

import httpx

from .chat import Chat
from .escape import stuck, stuck_line
from .fastprompt import compact_obs
from .fixes import failure_line, look_far_line
from .goals import clean, missing
from .ladder import Record
from .planwatch import TOLD, did_nothing
from .plans import LOCAL_SPECS, PlanStream, command_text, parse_command
from .prompts import summarize_result
from .senses import danger_lines
from .sim.specs import LEGEND

log = logging.getLogger("thinker")
MAX_PLAN = 6                                     # commands taken from one answer
KNOWN = {"tree", "rock", "berry bush", "bare bush", "thicket", "fruit tree", "ore", "mushroom", "workbench", "campfire",
         "water", "chest", "bed"}
AHEAD = 3                                        # the next call is always in flight unless this many commands wait
FAIL_WINDOW_S = 60.0
LOCAL = set(LOCAL_SPECS)                         # remember / forget: the harness keeps them, the game never sees them


def inv_after(obs, reply) -> dict:
    """The pack after an action: the observation's, plus the reply's changes."""
    inv = dict(obs["self"]["inventory"])
    for k, n in ((reply.get("result") or {}).get("changed") or {}).get("inventory", {}).items():
        inv[k] = inv.get(k, 0) + n
    return inv


def by_id(raw, obs):
    """'attack Mira' / 'give tusker ...': a name or kind in view stands for its id (the nearest one)."""
    if not isinstance(raw, str) or not obs:
        return raw
    w = raw.split()
    if len(w) >= 2 and w[0].lower() in ("attack", "give", "feed", "milk", "name") and not w[1].lstrip("#-").isdigit():
        who = w[1].lstrip("#").lower()
        e = next((e for e in obs.get("entities", []) if who in (str(e.get("name", "")).lower(), e["kind"])), None)
        if e:
            return " ".join([w[0], str(e["id"])] + w[2:])
    return raw


@dataclass
class Item:
    tool: str
    args: dict
    source: str                 # model | chat
    why: str = ""
    at: tuple | None = None     # where the plan expected to stand for a relative (dir) command
    epoch: int = 0
    plan: int = 0               # which answer it came from (repeats, failures of the same plan)


class Planner(Chat):
    # ------------------------------------------------------------- prompt
    def plan_messages(self, obs, pending=()) -> list[dict]:
        habit = [f"'{a}' x{n}" for a, n in Counter(self.recent).most_common(2) if n >= 4]
        fails = [v for k, v in self.lessons.items.items() if k.startswith("fail:")][-3:]
        me = obs["self"]
        self.record.seen(me["inventory"], time.time())
        obs = {**obs, "_seen": self.look_around(obs)}
        self.mem.save()
        names = [e["name"] for e in obs["entities"] if e["kind"] == "agent"]
        recall = self.mem.relevant(self.goal or "", (me["x"], me["y"]), names, k=2)
        people = [self.mem.person_text(n) for n in names[:2] if n in self.mem.people]
        if diary := self.mem.summary_text():
            recall = [f"diary: {diary[:160]}"] + recall
        if known := self.known_places(obs):
            recall = [known] + recall
        user = compact_obs(obs, goal=self.goal, record=self.record, heard=list(self.heard)[-3:], history=list(self.history),
                           pending=list(pending), notes=" ".join(filter(None, [self.mem.notes_text(),
                           (f"Your cards: {cards}" if (cards := self.cards.text(2)) else "")])), lessons=" | ".join(f[:90] for f in fails[-2:]),
                           danger=danger_lines(obs, self.body), recall=people + recall, goal_line=self.goal_line(obs),
                           habit=" ".join(filter(None, [stuck(obs, self.memory) and stuck_line(obs), self.fail_facts(),
                                                        f"Of your last 20 commands: {', '.join(habit)}." if habit else ""])))
        return [{"role": "system", "content": self.prefix}, {"role": "user", "content": user}]

    def look_around(self, obs) -> dict:
        """Where each kind of thing was last seen (from every view and look_far): the player's own map memory."""
        seen = self.memory["seen"] = self.mem.places         # (kept in memory.json: it survives a restart)
        g, r, me = obs["grid"], len(obs["grid"]) // 2, obs["self"]
        for j, row in enumerate(g):
            for i, ch in enumerate(row):
                if ch in "TMbvHQF*&^~":
                    seen[LEGEND.get(ch, ch).split(" (")[0].replace("_", " ")] = (me["x"] - r + i, me["y"] - r + j)
        return seen

    def known_places(self, obs) -> str:
        """Things not in view now and where they were last seen (the view, look_far), nearest first."""
        me, view = obs["self"], {LEGEND.get(c, c).split(" (")[0].replace("_", " ") for row in obs["grid"] for c in row}
        far = sorted((abs(p[0] - me["x"]) + abs(p[1] - me["y"]), k, p) for k, p in self.mem.places.items()
                     if k not in view and k in KNOWN)
        return ("seen before, out of view: " + ", ".join(f"{k} ({p[0]},{p[1]}) {d} tiles" for d, k, p in far[:5])) if far else ""

    def item(self, tool, args, source, why="") -> Item:
        return Item(tool, args, source, why, None, self.epoch)

    @property
    def record(self) -> Record:
        """What this player ever held and placed, and the milestones it reached (ladder.py), kept across runs."""
        if getattr(self, "_record", None) is None:
            self._record = Record(self.dir / "progress.json")
        return self._record

    def result_line(self, item, reply, obs) -> str:
        """The line the model reads back: what changed, or for a failure what would work instead (fixes.py)."""
        ok = bool(reply.get("ok"))
        self.record.did(item.tool, item.args, ok, time.time())
        self.check_goal({**obs, "self": {**obs["self"], "inventory": inv_after(obs, reply)}})
        if ok and item.tool == "attack":
            who = next((e.get("name") for e in obs["entities"] if e["id"] == item.args.get("target")), None)
            if who:
                self.mem.you_hit(who)
        if ok and item.tool == "place" and item.args.get("item") in ("workbench", "chest", "bed", "campfire", "door"):
            me = obs["self"]
            self.mem.add("place", f"you placed a {item.args['item']} ({item.args.get('dir')} of where you stood)",
                         None, (me["x"], me["y"]), str(obs.get("time", "")))
        if ok and item.tool == "look_far":
            return look_far_line(item.args, reply.get("result") or {}, obs["self"], self.memory.setdefault("seen", {}))
        if ok or reply.get("error") == "interrupted":
            return summarize_result(item.tool, item.args, reply)
        return failure_line(item.tool, item.args, reply, {**obs, "_seen": self.memory.get("seen", {})})

    def set_goal(self, goal, test):
        """A new goal and its own success test (goals.py), checked after every action."""
        if goal != self.goal and getattr(self, "goal_test", None):
            self.mem.add("goal", f"goal '{self.goal}' dropped for '{goal}'")
        self.goal, self.goal_test = goal, clean(test)
        self.goal_since, self.goal_t0 = dict(self.record.placed), time.time()

    def goal_line(self, obs) -> str:
        test = getattr(self, "goal_test", None)
        if not test:
            return f"Your goal '{self.goal}' has no done_when test." if self.goal else ""
        return (f"Your goal '{self.goal}' (done_when {json.dumps(test)}), for {int(time.time() - self.goal_t0)} s; "
                f"still missing: {'; '.join(missing(test, obs, self.record, self.goal_since)) or 'nothing'}.")

    def check_goal(self, obs):
        test = getattr(self, "goal_test", None)
        if test and not missing(test, obs, self.record, self.goal_since):
            secs = int(time.time() - self.goal_t0)
            self.tell(f"(goal '{self.goal}' done in {secs} s: {json.dumps(test)} holds)")
            self.mem.add("goal", f"goal '{self.goal}' done in {secs} s", None, (obs["self"]["x"], obs["self"]["y"]))
            self.decision_log.write(json.dumps({"kind": "goal", "t": time.time(), "goal": self.goal, "done": True,
                                                "test": test, "secs": secs}) + "\n")
            self.stats["goals_done"] += 1
            self.goal_test = None

    def replay(self) -> bool:
        """The plan said "repeat": with nothing else queued and its goal's test not yet holding, play it again."""
        rep = getattr(self, "repeat", None)
        test = getattr(self, "goal_test", None)
        if not rep or rep["left"] <= 0 or not test or not self.obs or not missing(test, self.obs, self.record, self.goal_since):
            return False
        rep["left"] -= 1
        self.queue.extend(Item(t, a, "model", f"repeat ({rep['left']} left)", None, self.epoch, rep["pid"]) for t, a in rep["cmds"])
        self.stats["repeats"] += 1
        return True

    def plan_failed(self, item, error):
        """A command of plan N failed: kept for the fact 'this plan failed k times in the last minute'; its repeat ends."""
        self.plan_fails = [f for f in getattr(self, "plan_fails", []) if time.time() - f[0] < FAIL_WINDOW_S]
        self.plan_fails.append((time.time(), tuple(getattr(self, "plans", {}).get(item.plan, ())), f"{command_text(item.tool, item.args)}: {error}"))
        if getattr(self, "repeat", None) and self.repeat["pid"] == item.plan:
            self.repeat = None

    def fail_facts(self) -> str:
        c = Counter(p for t, p, _ in getattr(self, "plan_fails", []) if p and time.time() - t < FAIL_WINDOW_S)
        last = {p: e for _, p, e in getattr(self, "plan_fails", [])}
        return " ".join(f"The plan '{', '.join(p)}' failed {n} times in the last minute (last: {last[p]})." for p, n in c.most_common(2) if n >= 2)

    def tell(self, line: str):
        """A line for the model's 'Done' list, once (not the same warning four times over)."""
        key = line[:60]
        for old in [h for h in self.history if h[:60] == key]:
            self.history.remove(old)
        self.history.append(line)

    def _body(self, messages, max_tokens, temperature, stream):
        return {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "stream": stream, **({"stream_options": {"include_usage": True}} if stream else {}),
                "chat_template_kwargs": {"enable_thinking": False}}

    # ------------------------------------------------------------ planner
    async def plan_loop(self, http):
        while True:
            while self.obs is None or sum(i.source == "model" for i in self.queue) >= AHEAD:
                await asyncio.sleep(0.05)
            try:
                self.planning = True
                await self.plan_once(http)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                log.warning("%s: model error %r", self.name, e)
                self.stats["model_error"] += 1
                await asyncio.sleep(1.0)
            finally:
                self.planning = False
            try:
                await self.reflect(http)                 # the diary, when due (one call, between plans)
            except Exception as e:                       # noqa: BLE001  (a diary failing never stops play)
                log.warning("%s: diary %r", self.name, e)

    async def plan_once(self, http):
        snap, epoch, hard = self.obs, self.epoch, getattr(self, "hard_aborts", 0)
        self.plan_id = pid = getattr(self, "plan_id", 0) + 1
        self.plans = getattr(self, "plans", {})
        mine = [i for i in self.queue if i.source == "model"]
        at = (snap["self"]["x"], snap["self"]["y"])       # where the plan was made: its first relative command is meant from here
        if any(i.tool in ("move_to", "step", "walk", "jump", "dig_down") for i in mine):
            at = None
        msgs = self.plan_messages(snap, [command_text(i.tool, i.args) for i in mine])
        ps, usage, t0 = PlanStream(), {}, time.monotonic()
        ttft = first_cmd = None
        n_ok = n_bad = 0
        stale = False
        plan: list[str] = []                     # this answer's commands, as written
        self.plans[pid] = plan
        parsed: list = []
        ahead = [command_text(i.tool, i.args) for i in ([self.current] if self.current else []) + mine]

        def take(raw):
            nonlocal at, n_ok, n_bad, first_cmd, stale
            c = parse_command(by_id(raw, self.obs))
            if c is None:
                n_bad += 1
                return
            tool, args = c
            text = command_text(tool, args)
            plan.append(text)
            parsed.append(c)
            if plan == ahead[:len(plan)]:       # the plan already running, asked again: it goes on, not from the start
                self.stats["dup_dropped"] += 1
                n_ok += 1
                return
            me = self.obs["self"]
            if tool == "move_to" and (args.get("x"), args.get("y")) == (me["x"], me["y"]) and not mine:
                self.stats["noop_dropped"] += 1  # already standing there
                n_ok += 1
                return
            self.queue.append(Item(tool, args, "model", ps.goal or "", at if "dir" in args else None, self.epoch, pid))
            n_ok += 1
            first_cmd = first_cmd or time.monotonic() - t0
            if tool in ("move_to", "step", "walk", "jump", "dig_down"):
                at = None                        # after its own move the plan trusts where that move ended
            self.new_plan.set()

        async with http.stream("POST", self.model_url, json=self._body(msgs, self.max_tokens, self.temperature, True),
                               timeout=120) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        ttft = ttft or time.monotonic() - t0
                    if getattr(self, "hard_aborts", 0) != hard:   # (a hit, a card, a chat: made before it, dropped;
                        stale = True                                 # a failed command: the plan streaming goes on)
                        break
                    for raw in ps.feed(piece):
                        if not stale:
                            take(raw)
                if stale or n_ok + n_bad >= MAX_PLAN:
                    break                        # (a long plan goes stale before it is done: stop decoding it)
        if not stale:
            for raw in ps.finish()[:max(0, MAX_PLAN - n_ok - n_bad)]:
                take(raw)
            if ps.goal and (ps.goal != self.goal or ps.done_when):
                self.set_goal(ps.goal, ps.done_when)
            if ps.note:
                self.add_note(ps.note)
            self.repeat = {"pid": pid, "cmds": parsed, "left": min(int(ps.repeat), 20)} if ps.repeat and parsed else None
            n = self.watch.arrived(tuple(plan), self.obs["self"]["inventory"]) if plan else 0
            if n >= TOLD:                        # asked again, and the pack is as it was when it came last time
                self.tell(did_nothing(tuple(plan), self.last_skip))
        if n_ok == 0 and not stale:
            self.stats["parse_fail"] += 1
            self.history.append("(your last answer had no command I could run: use the command forms shown)")
        rec = {"kind": "call", "t": time.time(), "latency_s": round(time.monotonic() - t0, 2),
               "ttft_s": round(ttft or 0, 2), "first_cmd_s": round(first_cmd or 0, 2), "n_cmds": n_ok, "bad": n_bad,
               "stale": stale, "goal": ps.goal, "usage": usage, "raw": ps.text[:400], **({"prompt": msgs[1]["content"]} if pid % 25 == 1 else {})}
        self.calls.append(rec)
        self.decision_log.write(json.dumps(rec) + "\n")
        log.info("%s [plan %.1fs] %s | %d cmds%s", self.name, rec["latency_s"], ps.goal or "-", n_ok,
                 " (dropped: stale)" if stale else "")
