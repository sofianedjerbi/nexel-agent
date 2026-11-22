"""LLM thinker harness: one ReAct-style decision per model call.

observe -> prompt (persona, notepad, grounded lessons, recent outcomes, observation) -> one JSON
tool call -> execute -> record the outcome. The model talks only to the local model server; the
harness talks only to the game server.

    uv run python -m megastructure.thinker --name Vex --persona "scavenger who trades"
"""

import argparse
import asyncio
import json
import logging
import random
import re
import time
from collections import Counter, deque
from pathlib import Path

import httpx

from .client import ApiError, Client
from .lessons import Lessons
from .policy import feedback, scripted
from .prompts import (
    FORMAT,
    RULES,
    recipes_text,
    render_obs,
    summarize_result,
    tools_text,
)
from .sim import TOOL_SPECS
from .tiles import FOOD

log = logging.getLogger("thinker")
MODEL_URL = "http://127.0.0.1:8080/v1/chat/completions"


class Thinker:
    def __init__(self, name, persona, token=None, game_url="ws://127.0.0.1:7777/agent", model_url=MODEL_URL,
                 model="default", data_dir=Path("agents"), temperature=0.7, max_tokens=300, decision_log=None):
        self.name, self.persona = name, persona
        self.token = token or f"thinker-{name}"
        self.game_url, self.model_url, self.model = game_url, model_url, model
        self.dir = data_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.notepad_path = self.dir / "notepad.md"
        self.lessons = Lessons(self.dir / "lessons.json")
        self.history: deque[str] = deque(maxlen=10)
        self.recent_actions: deque[str] = deque(maxlen=30)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.decision_log = open(decision_log or self.dir / "decisions.jsonl", "a")
        self.rng = random.Random(hash(name) & 0xFFFF)
        self.fallback_memory: dict = {}
        self.stats = Counter()

    # ------------------------------------------------------------ notepad
    def notepad(self) -> str:
        return self.notepad_path.read_text() if self.notepad_path.exists() else ""

    def add_note(self, line: str):
        line = line.strip().replace("\n", " ")[:200]
        if not line:
            return
        lines = [l for l in self.notepad().splitlines() if l.strip()]
        if line in lines:
            return
        lines.append(line)
        self.notepad_path.write_text("\n".join(lines[-30:]) + "\n")

    # ------------------------------------------------------------- prompt
    def messages(self, obs) -> list[dict]:
        habit = Counter(self.recent_actions).most_common(3)
        habit_txt = ", ".join(f"{a} x{n}" for a, n in habit if n >= 4)
        system = "\n\n".join([
            RULES,
            f"Your name is {self.name}. Who you are: {self.persona}",
            "Tools (one per turn):\n" + tools_text(),
            "Recipes: " + recipes_text(),
            "Food values: " + ", ".join(f"{k} +{v}" for k, v in FOOD.items()),
            FORMAT,
        ])
        user = "\n\n".join([
            "Your notepad (your own notes; may be wrong):\n" + (self.notepad() or "(empty)"),
            "Things you have verified from experience:\n" + self.lessons.text(),
            "Your last actions and their results:\n" + ("\n".join(self.history) or "(none yet)"),
            *( [f"You have been repeating yourself lately ({habit_txt}). Repetition is boring; "
                "unless it is clearly working, try something new."] if habit_txt else []),
            "What you perceive now:\n" + render_obs(obs),
            "Decide your next action. /no_think",   # (Qwen3: no long hidden reasoning; it used all of max_tokens and left no JSON)
        ])
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ---------------------------------------------------------- model I/O
    async def ask_model(self, http: httpx.AsyncClient, messages) -> tuple[dict | None, str, dict]:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        t0 = time.monotonic()
        r = await http.post(self.model_url, json=body, timeout=180)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        meta = {"latency_s": round(time.monotonic() - t0, 2), "usage": data.get("usage", {})}
        return parse_decision(text), text, meta

    # --------------------------------------------------------------- loop
    async def run(self, minutes: float | None = None):
        deadline = time.monotonic() + minutes * 60 if minutes else None
        async with httpx.AsyncClient() as http, Client(self.game_url) as game:
            await game.hello(self.name, self.token, self.persona[:200])
            log.info("%s joined", self.name)
            while deadline is None or time.monotonic() < deadline:
                try:
                    await self.step(http, game)
                except ApiError as e:
                    if e.code == "rate_limited":
                        await asyncio.sleep(2)
                    else:
                        raise

    async def step(self, http, game):
        obs = await game.observe()
        me = obs["self"]
        urgent = self.urgent(obs)
        source = "model"
        meta = {}
        raw = ""
        if urgent:
            tool, args, thought = urgent
            source = "reflex"
        else:
            try:
                decision, raw, meta = await self.ask_model(http, self.messages(obs))
            except (httpx.HTTPError, KeyError) as e:
                log.warning("%s: model error %s; using fallback", self.name, e)
                decision = None
            if decision is None:
                self.stats["parse_fail"] += 1
                tool, args, thought = scripted(obs, self.rng, self.persona, self.fallback_memory)
                source = "fallback"
            else:
                tool, args, thought = decision["tool"], decision.get("args") or {}, decision.get("thought", "")
                if decision.get("note"):
                    self.add_note(str(decision["note"]))
        reply = await game.act(tool, **args)
        feedback(self.fallback_memory, reply, obs["tick"])
        self.lessons.learn(tool, args, reply)
        line = summarize_result(tool, args, reply)
        self.history.append(f"[{obs['time']}] {line}")
        self.recent_actions.append(f"{tool}:{args.get('dir') or args.get('recipe') or args.get('item') or ''}")
        ok = bool(reply.get("ok"))
        self.stats[f"{source}:{'ok' if ok else 'fail'}"] += 1
        rec = {"t": time.time(), "tick": obs["tick"], "agent": self.name, "source": source, "pos": [me["x"], me["y"]],
               "hp": me["hp"], "hunger": me["hunger"], "thought": thought, "tool": tool,
               "args": args, "ok": ok, "error": reply.get("error"), "result": line, **meta}
        if source != "model" or not ok:
            rec["raw"] = raw[:500]
        self.decision_log.write(json.dumps(rec) + "\n")
        self.decision_log.flush()
        log.info("%s [%s %.1fs] %s | %s", self.name, source, meta.get("latency_s", 0), thought[:100], line[:140])

    def urgent(self, obs):
        """Reflexes that should not wait for a slow model: eat when starving, flee what bites."""
        me = obs["self"]
        inv = me["inventory"]
        if me["hunger"] <= 3:              # (food is 0 to 20)
            food = max((f for f in FOOD if inv.get(f)), key=lambda f: FOOD[f], default=None)
            if food:
                return "eat", {"item": food}, f"(reflex) starving, eating {food}"
        for e in obs["entities"]:
            if e["kind"] in ("tusker", "pincer", "strawman", "wyrm", "mossback") and abs(e["dx"]) + abs(e["dy"]) <= 2 and me["hp"] < 40:
                return "move_to", {"x": me["x"] - 4 * (e["dx"] > 0) + 4 * (e["dx"] <= 0),
                                   "y": me["y"] - 4 * (e["dy"] > 0) + 4 * (e["dy"] <= 0)}, "(reflex) fleeing"
        return None


def parse_decision(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(d, dict) and isinstance(d.get("tool"), str) and d["tool"] in TOOL_SPECS:
                        if not isinstance(d.get("args", {}), dict):
                            d["args"] = {}
                        return d
                    break
        start = text.find("{", start + 1)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--persona", default="a curious scavenger who wants to survive and build a home")
    p.add_argument("--game", default="ws://127.0.0.1:7777/agent")
    p.add_argument("--model-url", default=MODEL_URL)
    p.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")   # (the model server's id: "default" was a 404)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--data", default="agents")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(Thinker(a.name, a.persona, game_url=a.game, model_url=a.model_url, model=a.model,
                        data_dir=Path(a.data)).run(a.minutes))


if __name__ == "__main__":
    main()
