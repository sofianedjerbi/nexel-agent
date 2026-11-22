"""The thinker's model side: plan calls streamed into the queue (pipelined on the newest observation),
and the fast chat reply. Mixed into Thinker (thinker.py), whose body loop runs the queue.
"""

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass

import httpx

from .fastprompt import chat_messages, compact_obs
from .plans import PlanStream, command_text, parse_command

log = logging.getLogger("thinker")
MAX_PLAN = 6                                     # commands taken from one answer


@dataclass
class Item:
    tool: str
    args: dict
    source: str                 # model | filler | reflex | chat | align
    why: str = ""
    at: tuple | None = None     # where the plan expected to stand for a relative (dir) command
    epoch: int = 0


class Planner:
    # ------------------------------------------------------------- prompt
    def plan_messages(self, obs, pending=()) -> list[dict]:
        habit = [f"'{a}' x{n}" for a, n in Counter(self.recent).most_common(2) if n >= 4]
        notes = "; ".join(x for x in self.notepad().splitlines()[-4:] if x.strip())
        fails = [v for k, v in self.lessons.items.items() if k.startswith("fail:")][-3:]
        user = compact_obs(obs, goal=self.goal, heard=list(self.heard)[-3:], history=list(self.history), pending=list(pending),
                           notes=notes, lessons=" | ".join(f[:110] for f in fails),
                           habit=f"You keep doing {', '.join(habit)}: try something else." if habit else "")
        return [{"role": "system", "content": self.prefix}, {"role": "user", "content": user}]

    def _body(self, messages, max_tokens, temperature, stream):
        return {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "stream": stream, **({"stream_options": {"include_usage": True}} if stream else {}),
                "chat_template_kwargs": {"enable_thinking": False}}

    # ------------------------------------------------------------ planner
    async def plan_loop(self, http):
        while True:
            while self.obs is None or sum(i.source == "model" for i in self.queue) > self.refill_at:
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

    async def plan_once(self, http):
        snap, epoch = self.obs, self.epoch
        mine = [i for i in self.queue if i.source == "model"]
        at = (snap["self"]["x"], snap["self"]["y"])       # where the plan was made: its first relative command is meant from here
        if any(i.tool in ("move_to", "step", "walk", "jump", "dig_down") for i in mine):
            at = None
        msgs = self.plan_messages(snap, [command_text(i.tool, i.args) for i in mine])
        ps, usage, t0 = PlanStream(), {}, time.monotonic()
        ttft = first_cmd = None
        n_ok = n_bad = 0
        stale = False

        def take(raw):
            nonlocal at, n_ok, n_bad, first_cmd
            c = parse_command(raw)
            if c is None:
                n_bad += 1
                return
            tool, args = c
            self.queue.append(Item(tool, args, "model", ps.goal or "", at if "dir" in args else None, epoch))
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
                    if self.epoch != epoch:
                        stale = True
                        break
                    for raw in ps.feed(piece):
                        take(raw)
                if stale or n_ok + n_bad >= MAX_PLAN:
                    break                        # (a long plan goes stale before it is done: stop decoding it)
        if not stale:
            for raw in ps.finish()[:max(0, MAX_PLAN - n_ok - n_bad)]:
                take(raw)
            if ps.goal:
                self.goal = ps.goal
            if ps.note:
                self.add_note(ps.note)
        if n_ok == 0 and not stale:
            self.stats["parse_fail"] += 1
            self.history.append("(your last answer had no command I could run: use the command forms shown)")
        rec = {"kind": "call", "t": time.time(), "latency_s": round(time.monotonic() - t0, 2),
               "ttft_s": round(ttft or 0, 2), "first_cmd_s": round(first_cmd or 0, 2), "n_cmds": n_ok, "bad": n_bad,
               "stale": stale, "goal": ps.goal, "usage": usage, "raw": ps.text[:400]}
        self.calls.append(rec)
        self.decision_log.write(json.dumps(rec) + "\n")
        log.info("%s [plan %.1fs] %s | %d cmds%s", self.name, rec["latency_s"], ps.goal or "-", n_ok,
                 " (dropped: stale)" if stale else "")

    # --------------------------------------------------------------- chat
    def on_chat(self, speaker, text):
        if self.chat_task and not self.chat_task.done():
            self.chat_next = (speaker, text)
        else:
            self.chat_task = asyncio.create_task(self.chat_reply(speaker, text))

    async def chat_reply(self, speaker, text):
        while True:
            t0 = time.monotonic()
            try:
                r = await self.http.post(self.model_url, timeout=30, json=self._body(
                    chat_messages(self.name, self.persona, self.goal or "", speaker, text, list(self.heard)[-3:-1]),
                    40, 0.6, False))
                r.raise_for_status()
                line = (r.json()["choices"][0]["message"]["content"] or "").strip().splitlines()
                line = line[0].strip().strip('"').strip() if line else ""
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                log.warning("%s: chat error %r", self.name, e)
                line = ""
            if line:
                self.queue.appendleft(Item("say", {"text": line[:120]}, "chat", f"answering {speaker}", None, self.epoch))
                self.new_plan.set()
                self.calls.append({"kind": "chat", "latency_s": round(time.monotonic() - t0, 2)})
                self.decision_log.write(json.dumps({"kind": "chat", "t": time.time(), "to": speaker, "heard": text,
                                                    "said": line, "latency_s": round(time.monotonic() - t0, 2)}) + "\n")
            if not self.chat_next:
                return
            (speaker, text), self.chat_next = self.chat_next, None
