"""The live body's words, and the one speaker: every line the player says goes out through Voice.speak - its replies,
the planner's own line (rl/mind.py), the policy's say and ask (rl/live.py) - so one place keeps the pace of its talk
and the record of what it said. One small local model, its words and its thoughts in a lane each: a reply never waits
behind a thought (#11), and no thought starts while a reply is on its way. Every prompt (the words, the planner) is built on one text, rl/now.py Now.text(), what is happening to
it now; the words also get the talk lately with whoever spoke, its own lines in it. It answers what is said to it
(its name in the line, a talk it is in, the only one near it in earshot), not two others talking; lines heard while it
is answering join the next answer: a burst gets one reply. The model's reply - possibly nothing: it chooses - is
said. Every call is timed into latency.jsonl."""

import asyncio
import json
import re
import time
from collections import deque
from pathlib import Path

from nexel_worlds.sim.constants import SAY_R, sec
from . import acts
from nexel_worlds.tiles import RECIPES
from .facts import dist, named, needs, people
from .now import Now
from .social import TALK_GAP

# (one style for every line it says: the words here, the planner's line in rl/mind.py)
CHAT_STYLE = ("Type the way people really type in a game chat: mostly lowercase, capitals only now and then, short "
              "(often a few words, at most 15), fragments are fine, light punctuation, no exclamation spam, no emoji, "
              "never formal or robotic. You have your own things going on: friendly and easy, but not needy, you never "
              "fish for attention or keep the talk going for its own sake.")

WORDS_TOKENS, THOUGHT_TOKENS = 70, 200         # (a thought: its why, its JSON, and what it learned, if anything)
# The model to serve for the words and the planner (rl_runs/bench, 2026-11-23, with training at 4 envs): the 30B-A3B
# answered fastest (p90 0.6 s), never off-topic, and planned as well as any; the 8B broke the planner's JSON at 100
# tokens (19 of 20). It needs ~16 GB: with less free, the 4B (p90 0.8 s) is the fallback.
MODEL, FALLBACK_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit"
GAP_S = 2.0                          # at least this between two of its lines, whoever wants them said
OPEN_S = TALK_GAP / sec(1)           # a line of its own (not an answer): the sim's chat cooldown, unless spoken to since
TALK_S, THREAD_N = 90.0, 8           # a talk with someone: their lines and its own, the last 8, while it lasts (90 s)
NEAR_R = 6                           # the only person this near: what they say is to it


class Voice:
    def __init__(self, name, persona, url, http, log: Path, now: Now | None = None):
        self.name, self.persona, self.url, self.http, self.log = name, persona, url, http, Path(log)
        self.now = now or Now(name)
        self.lanes = {True: asyncio.Lock(), False: asyncio.Lock()}   # words, thoughts: each one at a time, never one behind the other
        self.words_waiting = 0
        self.inbox: list = []                # lines said to it, not answered yet: (who, text, heard_at)
        self.answering = False
        self.out = None                      # async (line) -> the game's result: the one way a line goes out (rl/live.py)
        self.threads: dict = {}              # person: deque of (t, speaker, text), its own lines in them
        self.last_said, self.heard_at = -1e9, -1e9
        self.to_me: dict = {}                # person: when they last said something to it
        self.mine: deque = deque(maxlen=6)   # its own last lines (t, text): none said twice in a row
        # (the fixed part first, the same every call: the server keeps its prefix cached)
        self.system = (f"You are {name}, a player in a survival sandbox game. Your character: {persona}\n{CHAT_STYLE}\n"
                       "The facts below are what is happening to you right now in the game. "
                       + acts.REPLY + ".")
        self.on_heard = lambda who, kind, item: None      # (rl/live.py: what each line said to it does: an ask is recorded)
        self.on_promise = lambda to, item, n: None        # (rl/live.py: its promise binds its body: rl/mind.py commit)
        self.asked_pack = False                           # (the last answer was to someone asking what it carries)

    async def post(self, messages, max_tokens, words: bool) -> tuple[str, dict]:
        """A call to the model. Words and thoughts each have a lane of their own: a reply never waits behind a thought
        in flight (#11: the server decodes the two together, mlx-lm's batching, as players.py starts it), and no
        thought starts while a reply waits or is being made."""
        t0 = time.monotonic()
        if words:
            self.words_waiting += 1
        else:
            while self.words_waiting:
                await asyncio.sleep(0.05)
        try:
            async with self.lanes[words]:
                t1 = time.monotonic()
                r = await self.http.post(self.url, timeout=40, json={"model": "default_model", "messages": messages,
                                                                     "max_tokens": max_tokens, "temperature": 0.7})
                d = r.json()
        finally:
            if words:
                self.words_waiting -= 1
        t2 = time.monotonic()
        u = d.get("usage") or {}
        out = d["choices"][0]["message"]["content"] or ""
        return out, {"queue_s": round(t1 - t0, 2), "llm_s": round(t2 - t1, 2), "prompt_tokens": u.get("prompt_tokens"),
                     "gen_tokens": u.get("completion_tokens"),
                     "tok_s": round((u.get("completion_tokens") or 0) / max(1e-3, t2 - t1), 1)}

    def thread(self, who: str) -> list:
        """The talk with `who` lately, oldest first: (speaker, text)."""
        t, q = time.monotonic(), self.threads.get(who) or ()
        return [(s, x) for at, s, x in q if t - at <= TALK_S]

    def note(self, who, speaker, text):
        self.threads.setdefault(who, deque(maxlen=THREAD_N)).append((time.monotonic(), speaker, text))

    def addressed(self, who: str, text: str) -> bool:
        """Is this line said to it: its name in it, a talk it is in with them (it spoke to them, or they to it, lately),
        or they are the only person near it in earshot? Two others talking is not."""
        if re.search(rf"\b{re.escape(self.name.lower())}\b", text.lower()):
            return True
        if any(s == self.name for s, _ in self.thread(who)) or time.monotonic() - self.to_me.get(who, -1e9) <= TALK_S:
            return True
        near = [p for p in people(self.now.obs) if dist(p) <= NEAR_R]
        return len(near) == 1 and near[0].get("name") == who

    def facts(self, heard: list, doing: str = "") -> str:
        """The lines said to it first (all of a burst), then what is happening now (the whole pack in it), the talk
        lately with whoever spoke, its own last lines, what it is after."""
        by = {}
        for who, text, _ in heard:
            by.setdefault(who, []).append(f'"{text}"')
        out = [f"{who} says to you: " + " then ".join(ts) for who, ts in by.items()] or ["Nobody is talking to you right now."]
        out.append(self.now.text())
        for who in by:
            talk = self.thread(who)[:-len(by[who])]          # (before the lines just heard)
            if talk:
                out.append(f"Your talk with {who} before that: " + " | ".join(
                    f'{"you" if s == self.name else s}: "{x}"' for s, x in talk))
        asked = [k for _, text, _ in heard for k in named(text, self.now.items) if k in RECIPES]
        for item in dict.fromkeys(asked).keys() if heard else ():      # (a thing named: what the game says it takes)
            out.append(f"What {item} takes (the game's recipe): {needs(item, self.now.obs)}.")
        if self.mine:
            out.append("Your own last lines: " + " | ".join(f'"{x}"' for _, x in self.mine) + " (do not say one again).")
        if doing:
            out.append(f"You are after: {doing}.")
        return "\n".join(out)

    async def line(self, heard: list, doing: str = "", kind="reply") -> str:
        """The model's line to the lines `heard` (or, none, to start talk); '' when it says nothing."""
        user = self.facts(heard, doing) + "\n/no_think"
        try:
            out, t = await self.post([{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                                     WORDS_TOKENS, words=True)
        except Exception as e:
            out, t = "", {"error": repr(e)[:80]}
        raw, (out, labels, promise) = out, acts.read(out, len(heard), self.now.items)
        for (who, _, _), (act, item) in zip(heard, labels):
            self.on_heard(who, act, item)
        self.asked_pack = any(k == "pack" for k, _ in labels)
        out = next((ln.strip() for ln in out.splitlines() if ln.strip()), "").strip('"')[:120]     # (one line)
        if out.lower() in ("", "nothing", "(nothing)", "...", "-"):
            out = ""
        claim = acts.deed(raw, self.now.items)
        if out and claim and not self.now.backs(*claim, heard[-1][0] if heard else None):
            self.write({"kind": "dropped", "said": out, "why": f"claims it {' '.join(claim)}: no such deed"})
            out = ""
        if out and promise and heard:
            self.on_promise(heard[-1][0], *promise)
        t.update(kind=kind, to=heard[-1][0] if heard else "", heard=" | ".join(h[1] for h in heard)[:120], said=out,
                 acts=[" ".join(x for x in lab if x) for lab in labels], promise=promise,
                 t=round(time.time(), 2), since_heard_s=round(time.monotonic() - heard[0][2], 2) if heard else None)
        self.write(t)
        return out

    def hear(self, who: str, text: str, doing=lambda: ""):
        """Someone spoke. Into the talk with them; said to it, into the inbox: one answer at a time takes all that came
        in meanwhile."""
        self.note(who, who, text)
        if not self.addressed(who, text):
            return
        self.heard_at = self.to_me[who] = time.monotonic()
        self.inbox.append((who, text, self.heard_at))
        if not self.answering:
            self.answering = True
            asyncio.get_running_loop().create_task(self.answer(doing))

    async def speak(self, line: str, to: str | None = None, reply: bool = False):
        """The one way a line goes out. Held back (None) when it is too soon after its last line, when it is a line
        of its own and it was not spoken to since its last one and OPEN_S has not passed, or when it said it just now;
        else sent (the game's result)."""
        line, t = (line or "").strip(), time.monotonic()
        spoken_to = reply and self.heard_at > self.last_said       # (an answer to what was said since its last line goes at once, #11)
        if reply and not spoken_to and t - self.last_said < GAP_S:      # (another answer waits its turn; a line of its own is dropped)
            await asyncio.sleep(GAP_S - (t - self.last_said))
            t = time.monotonic()
        if not line or self.out is None or (t - self.last_said < GAP_S and not spoken_to):
            return None
        if not reply and t - self.last_said < OPEN_S and self.heard_at <= self.last_said:
            return None
        if any(x.lower() == line.lower() for _, x in self.mine):
            return None
        if acts.pack_dump(line, self.now.items, (self.now.obs.get("self") or {}).get("inventory")) and not (reply and self.asked_pack):
            self.write({"kind": "dropped", "said": line, "why": "reads out the pack"})
            return None
        self.last_said = t
        self.mine.append((t, line))
        for who in [to] if to else [p.get("name") for p in people(self.now.obs) if dist(p) <= SAY_R]:   # (who heard it: the game's)
            self.note(who, self.name, line)
        return await self.out(line)

    def news(self, what: str, doing=lambda: ""):
        """Something it may want to say a word about (someone it knows just joined): the model is asked, and may well
        say nothing; the line keeps the pace of a line of its own."""
        if self.answering:
            return

        async def go():
            out = await self.line([], f"{doing()}. Just now: {what}".lstrip(". "), kind="news")
            if out:
                await self.speak(out)
        asyncio.get_running_loop().create_task(go())

    async def answer(self, doing):
        try:
            while self.inbox:
                heard, self.inbox = self.inbox, []
                out = await self.line(heard, doing())
                if out and await self.speak(out, heard[-1][0], reply=True) is not None:
                    self.write({"kind": "sent", "to": heard[-1][0], "said": out, "total_s": round(time.monotonic() - heard[0][2], 2)})
        finally:
            self.answering = False

    def write(self, rec: dict):
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
