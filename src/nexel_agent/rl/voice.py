"""The live body's words: one small local model, one request at a time, chat lines first (a planner thought waits
for them). Being spoken to is a salient event: the line heard goes to the model at once, with a short facts-only
prompt, and the model's reply - possibly nothing: it chooses - is said. The policy still chooses when to start talk
by itself (rl/live.py). Every call is timed into latency.jsonl."""

import asyncio
import json
import re
import time
from collections import deque
from pathlib import Path

from ..chat import CHAT_STYLE
from ..fastprompt import chat_parse

WORDS_TOKENS, THOUGHT_TOKENS = 30, 100
# The model to serve for the words and the planner (rl_runs/bench, 2026-11-23, with training at 4 envs): the 30B-A3B
# answered fastest (p90 0.6 s), never off-topic, and planned as well as any; the 8B broke the planner's JSON at 100
# tokens (19 of 20). It needs ~16 GB: with less free, the 4B (p90 0.8 s) is the fallback.
MODEL, FALLBACK_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit"
SIGHTS = ("water", "deep_water", "tree", "rock", "ore", "thicket", "berry_bush", "fruit_tree", "campfire", "workbench",
          "bed", "door", "chest", "well", "cave_floor", "hole", "ladder", "stone_wall", "wood_wall", "fence")


def compass(dx, dy) -> str:
    d = max(abs(dx), abs(dy))
    if d <= 1:
        return "right here"
    ns = "S" if dy > d / 2.4 else "N" if dy < -d / 2.4 else ""
    ew = "E" if dx > d / 2.4 else "W" if dx < -d / 2.4 else ""
    return f"{d} tiles {ns}{ew}"


def places(obs: dict, signs: dict) -> str:
    """The real places it knows of: signs it has read (where, what they say), its spawn, what is in sight now."""
    me = obs.get("self") or {}
    x, y = me.get("x", 0), me.get("y", 0)
    out = [f'a sign "{t[:40]}" {compass(sx - x, sy - y)}' for (sx, sy), t in list(signs.items())[-3:]]
    sp = me.get("spawn")
    if sp and max(abs(sp[0] - x), abs(sp[1] - y)) > 3:
        out.append(f"your spawn point {compass(sp[0] - x, sp[1] - y)}")
    legend = obs.get("legend") or {}
    seen = {legend.get(ch, "") for row in obs.get("grid") or [] for ch in row}
    sights = [s.replace("_", " ") for s in SIGHTS if s in seen]
    if sights:
        out.append("in sight: " + ", ".join(sights[:6]))
    if me.get("in_cave"):
        out.append("you are in a cave")
    return "; ".join(out) or "none"


class Voice:
    def __init__(self, name, persona, url, http, log: Path, items=()):
        self.name, self.persona, self.url, self.http, self.log = name, persona, url, http, Path(log)
        self.lock = asyncio.Lock()
        self.words_waiting = 0
        self.items = tuple(items)            # the things a pack can hold, to tell which a line names
        self.chat = deque(maxlen=5)          # the last lines said near it (its own too): (who, text)
        # (the fixed part first, the same every call: the server keeps its prefix cached)
        self.system = (f"You are {name}, a player in a survival sandbox game. Your character: {persona}\n{CHAT_STYLE}\n"
                       "Reply with only your chat line, to what was just said to you. If you would rather not answer, reply with nothing.")

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

    def facts(self, speaker, text, me: dict, doing: str, note: str = "", acting: str = "", where: str = "") -> str:
        """What a reply needs, the line first; of the pack only the items the line names, with their counts."""
        inv = me.get("inventory") or {}
        low = (text or "").lower()
        words = set(re.findall(r"[a-z]+", low))
        named = [k for k in self.items if k.replace("_", " ") in low if " " in k.replace("_", " ") or k in words or k + "s" in words]
        out = [f'{speaker} says to you: "{text}"' if text else f"{speaker} is near you; nobody has said anything yet."]
        before = [f"{w}: {t}" for w, t in list(self.chat)[:-1]] if text else [f"{w}: {t}" for w, t in self.chat]
        if before:
            out.append("Said just before: " + " | ".join(before[-2:]))
        if note:
            out.append(f"What you know of {speaker}: {note}")
        now = "; ".join(p for p in (acting and f"your last action: {acting}", doing) if p)
        if now:
            out.append(now[0].upper() + now[1:] + ".")
        if where:
            out.append(f"Places you know: {where}")
        if named:
            out.append("You carry " + ", ".join(f"{inv.get(k, 0)} {k}" for k in named[:4]) + ".")
        if me.get("hp", 20) <= 10 or me.get("hunger", 20) <= 6:
            out.append(f"(health {me.get('hp')}/20, food {me.get('hunger')}/20)")
        return "\n".join(out)

    async def line(self, speaker: str, text: str, me: dict, doing: str, heard_at: float | None = None, kind="reply",
                   note: str = "", acting: str = "", where: str = "") -> str:
        """The model's line to `speaker` who said `text` (or, `text` empty, to start talk); '' when it says nothing."""
        if text:
            self.chat.append((speaker, text))
        user = self.facts(speaker, text, me, doing, note, acting, where) + "\n/no_think"
        try:
            out, t = await self.post([{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                                     WORDS_TOKENS, words=True)
        except Exception as e:
            out, t = "", {"error": repr(e)[:80]}
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        if out.startswith("{"):
            out = chat_parse(out)[0]
        out = next((ln.strip() for ln in out.splitlines() if ln.strip()), "").strip('"')[:120]     # (one line)
        if out.lower() in ("", "nothing", "(nothing)", "...", "-"):
            out = ""
        if out:
            self.chat.append((self.name, out))
        t.update(kind=kind, to=speaker, heard=text[:80], said=out, t=round(time.time(), 2),
                 since_heard_s=round(time.monotonic() - heard_at, 2) if heard_at else None)
        self.write(t)
        return out

    def write(self, rec: dict):
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
