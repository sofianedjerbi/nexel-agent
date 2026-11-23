"""The live body's words: one small local model, one request at a time, chat lines first (a planner thought waits
for them). Every prompt (the words, the planner: rl/mind.py) is built on one text, Now.text(): what is happening to
it now - health and food and how they went, who is near and what they did, what its body did, what it got or lost,
what was said. Lines heard while it is answering join the next answer: a burst gets one reply. The model's reply -
possibly nothing: it chooses - is said. The policy still chooses when to start talk by itself (rl/live.py). Every
call is timed into latency.jsonl."""

import asyncio
import json
import math
import re
import time
from collections import Counter, deque
from itertools import groupby
from pathlib import Path

from .facts import HOSTILE, Attack, Craft, Give, Say, compass, counted, dist, events, is_creature, named, people, plain, told

# (one style for every line it says: the words here, the planner's line in rl/mind.py)
CHAT_STYLE = ("Type the way people really type in a game chat: mostly lowercase, capitals only now and then, short "
              "(often a few words, at most 15), fragments are fine, light punctuation, no exclamation spam, no emoji, "
              "never formal or robotic. You have your own things going on: friendly and easy, but not needy, you never "
              "fish for attention or keep the talk going for its own sake.")

WORDS_TOKENS, THOUGHT_TOKENS = 30, 100
# The model to serve for the words and the planner (rl_runs/bench, 2026-11-23, with training at 4 envs): the 30B-A3B
# answered fastest (p90 0.6 s), never off-topic, and planned as well as any; the 8B broke the planner's JSON at 100
# tokens (19 of 20). It needs ~16 GB: with less free, the 4B (p90 0.8 s) is the fallback.
MODEL, FALLBACK_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit"
SIGHTS = ("water", "deep_water", "tree", "rock", "ore", "thicket", "berry_bush", "fruit_tree", "campfire", "workbench",
          "bed", "door", "chest", "well", "cave_floor", "hole", "ladder", "stone_wall", "wood_wall", "fence")
WINDOW_S = 30.0                       # "lately": what it saw in the last 30 s; an ask not met in that time was not met
WORDS = {"N": "north", "S": "south", "E": "east", "W": "west", "noop": "wait"}


def places(obs: dict, signs: dict) -> str:
    """The real places it knows of: signs it has read (where, what they say), its spawn, what is in sight now."""
    me = obs.get("self") or {}
    x, y = me.get("x", 0), me.get("y", 0)
    out = [f'a sign "{t[:40]}" {compass(sx - x, sy - y)}' for (sx, sy), t in list(signs.items())[-3:]]
    sp = me.get("spawn")
    if sp and math.hypot(sp[0] - x, sp[1] - y) > 3:
        out.append(f"your spawn point {compass(sp[0] - x, sp[1] - y)}")
    legend = obs.get("legend") or {}
    seen = {legend.get(ch, "") for row in obs.get("grid") or [] for ch in row}
    sights = [s.replace("_", " ") for s in SIGHTS if s in seen]
    if sights:
        out.append("in sight: " + ", ".join(sights[:6]))
    if me.get("in_cave"):
        out.append("you are in a cave")
    return "; ".join(out) or "none"


def runs(words) -> str:
    """'a, a, b' -> 'a x2, b' (in order)."""
    return ", ".join(w + (f" x{n}" if (n := len(list(g))) > 1 else "") for w, g in groupby(words))


class Now:
    """What is happening to the player now, in plain facts: the last observation and what it saw in the last WINDOW_S.
    Fed each observation (see) and each action of its body (did); read by every prompt (text). It also keeps, per
    person, a record of what happened between them - outcomes only, counted: asked, given, not given, hit, helped -
    in the planner's memory (rl/mind.py relations[name]["record"])."""

    def __init__(self, name: str, items=()):
        self.name, self.obs, self.signs, self.items = name, {}, {}, tuple(items)   # items: what a pack can hold
        self.relations: dict = {}            # name: {ally, record} (rl/mind.py keeps it across sessions)
        self.asks: dict = {}                 # (who asked, whom): (item, t), open for WINDOW_S
        self.fought = -1e9                   # when a creature last hit it
        self.past: deque = deque()           # (t, event) within the window
        self.body: deque = deque()           # (t, hp, food, inventory): the oldest is the window's start
        self.acts: deque = deque(maxlen=6)

    def see(self, obs: dict):
        t, me = time.monotonic(), obs.get("self") or {}
        self.obs = obs
        for s in obs.get("signs", []):
            self.signs[(me.get("x", 0) + s["dx"], me.get("y", 0) + s["dy"])] = str(s.get("text", ""))
        for ev in events(obs):
            self.past.append((t, ev))
            self.outcome(ev, t)
        for (asker, whom), (item, t0) in [kv for kv in self.asks.items() if t - kv[1][1] > WINDOW_S]:
            del self.asks[(asker, whom)]
            self.remember(whom, f"did not give you the {item} you asked for") if asker == self.name else \
                self.remember(asker, f"you did not give them the {item} they asked for")
        self.body.append((t, me.get("hp"), me.get("hunger"), dict(me.get("inventory") or {})))
        while self.past and t - self.past[0][0] > WINDOW_S:
            self.past.popleft()
        while len(self.body) > 1 and t - self.body[1][0] >= WINDOW_S:
            self.body.popleft()

    def remember(self, who, what: str, n=1):
        if who and who != self.name and not is_creature(who):     # (people only)
            rec = self.relations.setdefault(who, {}).setdefault("record", {})
            rec[what] = rec.get(what, 0) + n

    def outcome(self, ev, t: float):
        me = self.name
        match ev:
            case Attack(by=a, target=tgt) if tgt == me:
                self.remember(a, "hit you")
                self.fought = t if is_creature(a) else self.fought
            case Attack(by=a, target=tgt) if a == me:
                self.remember(tgt, "you hit them")
            case Attack(by=a, target=tgt) if is_creature(tgt) and t - self.fought <= WINDOW_S:
                self.remember(a, "fought a creature that was fighting you")
            case Give(src=src, to=to) if me in (src, to):
                for k, n in ev.items.items():
                    self.remember(src if to == me else to, f"gave you {k}" if to == me else f"you gave them {k}", n)
                    if self.asks.get((to, src), ("",))[0] == k:
                        del self.asks[(to, src)]
            case Say(src=src) if src != me and (item := next(iter(named(ev.text, self.items)), None)):
                self.ask(src, me, item, t)

    def ask(self, who, whom, item, t=None):
        """`who` asked `whom` for `item` (its own asks: the policy's ask action, rl/live.py)."""
        whom = whom or next((b.get("name") for b in people(self.obs)), None)
        if who and whom:
            self.asks[(who, whom)] = (item, time.monotonic() if t is None else t)
            self.remember(who if who != self.name else whom, f"asked you for {item}" if who != self.name else f"you asked them for {item}")

    def person(self, name) -> str:
        """What happened between it and `name` so far, and whether it calls them an ally."""
        rel = self.relations.get(name) or {}
        rec = [f"{w} x{n:g}" if n != 1 else w for w, n in (rel.get("record") or {}).items()]
        ally = {True: "your ally", False: "not your ally"}.get(rel.get("ally"))
        return "; ".join(([ally] if ally else []) + rec) or "nothing between you yet"

    def did(self, action: str):
        self.acts.append(" ".join(WORDS.get(w, w) for w in action.split("_")))

    def who(self, name) -> str:
        return "you" if name == self.name else plain(name)

    def event(self, ev) -> str | None:
        """An event of the window in words (rl/facts.py told); chat has its own line, and of crafts only its own count."""
        return None if isinstance(ev, Say) or (isinstance(ev, Craft) and ev.name != self.name) else told(ev, self.who)

    def text(self) -> str:
        obs, now = self.obs, time.monotonic()
        me = obs.get("self") or {}
        t0, hp0, food0, inv0 = self.body[0] if self.body else (now, me.get("hp"), me.get("hunger"), {})
        ago = f"{now - t0:.0f} s ago"
        out = [f"You: health {me.get('hp')}/20" + (f" (was {hp0} {ago})" if hp0 != me.get("hp") else "")
               + f", food {me.get('hunger')}/20" + (f" (was {food0} {ago})" if food0 != me.get("hunger") else "")
               + (f". {obs['time']}." if obs.get("time") else ".")]
        beings = sorted(obs.get("entities", []), key=dist)
        near = []
        for kind in dict.fromkeys(b["kind"] for b in beings if b.get("kind") != "agent"):
            d = [round(dist(b)) for b in beings if b["kind"] == kind]
            near.append(f"{len(d)} {kind}{' (foe)' if kind in HOSTILE else ''} {d[0]}" + (f"-{d[-1]}" if d[-1] != d[0] else "")
                        + " tiles away")
        for b in people(obs):
            near.append(f"{b.get('name')} {compass(b['dx'], b['dy'])}, health {b.get('hp')}, holding {b.get('holding') or 'nothing'}, "
                        f"doing: {b.get('doing') or 'idle'} (so far: {self.person(b.get('name'))})")
        out.append("Near you: " + ("; ".join(near) if near else "nobody, no creature") + ".")
        seen = Counter(p for _, e in self.past if (p := self.event(e)))
        if seen:
            out.append(f"In the last {WINDOW_S:.0f} s: " + "; ".join(p + (f" x{n}" if n > 1 else "") for p, n in seen.items()) + ".")
        if self.acts:
            out.append("Your body's last actions: " + runs(self.acts) + ".")
        inv = me.get("inventory") or {}
        got = {k: n - inv0.get(k, 0) for k, n in inv.items() if n > inv0.get(k, 0)}
        lost = {k: n - inv.get(k, 0) for k, n in inv0.items() if n > inv.get(k, 0)}
        if got or lost:
            out.append(f"Since {ago}: " + "; ".join(p for p in (got and "you got " + counted(got),
                                                                   lost and "you used or lost " + counted(lost)) if p) + ".")
        chat = [told(ev, self.who) for _, ev in self.past if isinstance(ev, Say)]
        if chat:
            out.append("Said near you lately: " + " | ".join(chat[-4:]))
        out.append("Places you know: " + places(obs, self.signs))
        return "\n".join(out)


class Voice:
    def __init__(self, name, persona, url, http, log: Path, now: Now | None = None):
        self.name, self.persona, self.url, self.http, self.log = name, persona, url, http, Path(log)
        self.now = now or Now(name)
        self.lock = asyncio.Lock()
        self.words_waiting = 0
        self.inbox: list = []                # lines heard, not answered yet: (who, text, heard_at)
        self.answering = False
        # (the fixed part first, the same every call: the server keeps its prefix cached)
        self.system = (f"You are {name}, a player in a survival sandbox game. Your character: {persona}\n{CHAT_STYLE}\n"
                       "The facts below are what is happening to you right now in the game. "
                       "Reply with only your chat line. If you would rather say nothing, reply with nothing.")

    async def post(self, messages, max_tokens, words: bool) -> tuple[str, dict]:
        """A call to the model; chat lines go before thoughts."""
        t0 = time.monotonic()
        if words:
            self.words_waiting += 1
        else:
            while self.words_waiting:
                await asyncio.sleep(0.05)
        try:
            async with self.lock:
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

    def facts(self, heard: list, doing: str = "") -> str:
        """The lines heard first (all of a burst), then what is happening now; of the pack only the items the lines name."""
        by = {}
        for who, text, _ in heard:
            by.setdefault(who, []).append(f'"{text}"')
        out = [f"{who} says to you: " + " then ".join(ts) for who, ts in by.items()] or ["Nobody is talking to you right now."]
        out.append(self.now.text())
        if doing:
            out.append(f"You are after: {doing}.")
        inv = (self.now.obs.get("self") or {}).get("inventory") or {}
        asked = named(" ".join(t for _, t, _ in heard), self.now.items)
        if asked:
            out.append("You carry " + ", ".join(f"{inv.get(k, 0)} {k}" for k in asked[:4]) + ".")
        return "\n".join(out)

    async def line(self, heard: list, doing: str = "", kind="reply") -> str:
        """The model's line to the lines `heard` (or, none, to start talk); '' when it says nothing."""
        user = self.facts(heard, doing) + "\n/no_think"
        try:
            out, t = await self.post([{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                                     WORDS_TOKENS, words=True)
        except Exception as e:
            out, t = "", {"error": repr(e)[:80]}
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        if out.startswith("{"):                      # (asked for a line, it answered in JSON: its "say")
            try:
                out = str(json.loads(out).get("say") or "")
            except (ValueError, AttributeError):
                pass
        out = next((ln.strip() for ln in out.splitlines() if ln.strip()), "").strip('"')[:120]     # (one line)
        if out.lower() in ("", "nothing", "(nothing)", "...", "-"):
            out = ""
        t.update(kind=kind, to=heard[-1][0] if heard else "", heard=" | ".join(h[1] for h in heard)[:120], said=out,
                 t=round(time.time(), 2), since_heard_s=round(time.monotonic() - heard[0][2], 2) if heard else None)
        self.write(t)
        return out

    def hear(self, who: str, text: str, say, doing=lambda: ""):
        """Someone spoke: into the inbox; one answer at a time takes all that came in meanwhile (`say` sends a line)."""
        self.inbox.append((who, text, time.monotonic()))
        if not self.answering:
            self.answering = True
            asyncio.get_running_loop().create_task(self.answer(say, doing))

    async def answer(self, say, doing):
        try:
            while self.inbox:
                heard, self.inbox = self.inbox, []
                out = await self.line(heard, doing())
                if out:
                    await say(out)
                    self.write({"kind": "sent", "to": heard[-1][0], "said": out, "total_s": round(time.monotonic() - heard[0][2], 2)})
        finally:
            self.answering = False

    def write(self, rec: dict):
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
