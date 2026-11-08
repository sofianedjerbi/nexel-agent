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
from .policy import feedback, scripted
from .sim import TOOL_SPECS
from .tiles import FOOD, RECIPES

log = logging.getLogger("thinker")
MODEL_URL = "http://127.0.0.1:8080/v1/chat/completions"

RULES = """You are a person living on an endless archipelago of wild green islands.
Nobody rules here. There are no laws and no safe zones: anything can be built, dug, claimed, traded,
stolen or destroyed. Others like you live here too; they may help you, trade with you, or rob you.

World facts:
- Top-down tile grid. North = up (y decreases). Coordinates are absolute (x, y).
- You get hungry; at hunger 0 you lose health. Eat berries, mushrooms, eggs, milk, cooked meat or bread.
- Energy drops as you work; rest to recover (faster by a campfire or with walls around you).
- Dig adjacent things to harvest them: trees -> wood, berry bushes -> berries + seeds, thickets -> fiber,
  rocks -> stone (slow by hand), ore rocks -> ore (needs a pickaxe), mushrooms -> mushroom.
- Digging bare ground tills it. Plant seeds in tilled soil; after a few minutes they ripen into wheat.
- A workbench (4 wood) unlocks tools: axe (faster trees), pickaxe (rock, ore), spear (weapon).
- Chickens and cows wander around; they are food. At night, hostile glooms hunt people.
- Death drops everything you carry where you fell. You come back, empty-handed, somewhere else.
- Everyone sees what you say nearby; shouting carries far but is tiring.
"""

FORMAT = """Reply with ONE JSON object and nothing else:
{"thought": "<one or two short sentences: what you see and why this action>",
 "tool": "<tool name>", "args": {...},
 "note": "<optional: a line to add to your notepad (plans, people, places). Omit if nothing new>"}"""


def tools_text() -> str:
    lines = []
    for name, spec in TOOL_SPECS.items():
        args = ", ".join(f"{k}: {v}" for k, v in spec["args"].items()) or "-"
        lines.append(f"- {name}({args}): {spec['desc']}")
    return "\n".join(lines)


def recipes_text() -> str:
    out = []
    for name, r in RECIPES.items():
        ins = " + ".join(f"{n} {k}" for k, n in r.inputs.items())
        out.append(f"{name} = {ins}" + (f" [near {r.station}]" if r.station else ""))
    return "; ".join(out)


def render_obs(obs: dict) -> str:
    me = obs["self"]
    grid = obs["grid"]
    r = len(grid) // 2
    # column/row offsets help the model convert grid cells to absolute coordinates
    header = "    " + "".join(str((me["x"] - r + i) % 10) for i in range(len(grid[0])))
    rows = [f"{me['y'] - r + j:>4}" + row for j, row in enumerate(grid)]
    lines = [f"Time: {obs['time']}. You are '@' at ({me['x']},{me['y']}), standing on {me['standing_on']}"
             + (", sheltered" if me["sheltered"] else "") + ".",
             f"Health {me['hp']}/100, hunger {me['hunger']}/100 (low = starving), energy {me['energy']}/100, "
             f"deaths so far {me['deaths']}.",
             "Inventory: " + (", ".join(f"{k} x{v}" for k, v in me["inventory"].items()) or "nothing") + ".",
             "Map (x digits on top are the last digit of x; y on the left):", header, *rows,
             "Legend: " + ", ".join(f"{k}={v}" for k, v in obs["legend"].items())]
    adj = []
    for d, (dx, dy) in {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items():
        adj.append(f"{d}={obs['legend'].get(grid[r + dy][r + dx], grid[r + dy][r + dx])}")
    lines.append("Adjacent: " + ", ".join(adj))
    if obs["entities"]:
        ents = []
        for e in obs["entities"][:10]:
            pos = f"({me['x'] + e['dx']},{me['y'] + e['dy']})"
            if e["kind"] == "agent":
                ents.append(f"{e['name']} [id {e['id']}] at {pos}, hp {e['hp']}, holding {e['holding']}, {e['doing']}")
            else:
                ents.append(f"{e['kind']} [id {e['id']}] at {pos}, hp {e['hp']}")
        lines.append("Nearby: " + "; ".join(ents))
    if obs["items"]:
        lines.append("Items on the ground: " + "; ".join(
            f"({me['x'] + i['dx']},{me['y'] + i['dy']}): " + ", ".join(f"{k} x{v}" for k, v in i["items"].items())
            for i in obs["items"][:6]))
    if obs["signs"]:
        lines.append("Signs: " + "; ".join(f"({me['x'] + s['dx']},{me['y'] + s['dy']}) {s['kind']}: \"{s['text']}\""
                                           for s in obs["signs"][:5]))
    heard = [e for e in obs["events"] if e["type"] in ("say", "shout", "give", "attack", "death", "sign")]
    if heard:
        lines.append("Recently: " + "; ".join(describe_event(e) for e in heard[-8:]))
    return "\n".join(lines)


def describe_event(e: dict) -> str:
    t = e["type"]
    if t in ("say", "shout"):
        return f"{e['from']} {'shouted' if t == 'shout' else 'said'} \"{e['text']}\""
    if t == "give":
        return f"{e['from']} gave {e['to']} {', '.join(f'{v} {k}' for k, v in e['items'].items())}"
    if t == "attack":
        return f"{e['attacker']} hit {e['target']} for {e['damage']}"
    if t == "death":
        return f"{e['name']} died ({e['cause']})"
    if t == "sign":
        return f"{e['by']} wrote \"{e['text']}\""
    return t


def summarize_result(tool, args, reply) -> str:
    a = json.dumps(args, separators=(",", ":"))
    if reply.get("ok"):
        res = reply["result"]
        ch = res.get("changed", {})
        bits = []
        if "inventory" in ch:
            bits.append(", ".join(f"{'+' if v > 0 else ''}{v} {k}" for k, v in ch["inventory"].items()))
        if "pos" in ch:
            bits.append(f"now at ({ch['pos'][0]},{ch['pos'][1]})")
        for k in ("text", "made", "target", "killed", "chest", "rows_near_to_far"):
            if k in res:
                bits.append(f"{k}={json.dumps(res[k])[:300]}")
        return f"{tool}{a} -> ok ({res['ticks']} ticks){': ' + '; '.join(bits) if bits else ''}"
    return f"{tool}{a} -> FAILED {reply.get('error')}: {reply.get('detail')}"


class Lessons:
    """Knowledge derived only from observed outcomes, deduplicated by key."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def learn(self, tool, args, reply):
        if reply.get("ok"):
            res = reply["result"]
            got = res.get("changed", {}).get("inventory", {})
            if tool == "dig" and got:
                tiles = res.get("changed", {}).get("tiles", [])
                self._add(f"dig:{','.join(sorted(got))}", f"Digging gave {', '.join(f'{v} {k}' for k, v in got.items())}"
                          + (f" (the tile became {tiles[0]['block'] if tiles[0]['block'] != 'none' else tiles[0]['ground']})"
                             if tiles else ""))
            elif tool == "craft":
                self._add(f"craft:{args.get('recipe')}", f"Crafting {args.get('recipe')} works: "
                          f"{', '.join(f'{v:+d} {k}' for k, v in got.items())}")
            elif tool == "eat":
                ch = res.get("changed", {})
                self._add(f"eat:{args.get('item')}", f"Eating {args.get('item')} changed hunger by {ch.get('hunger', 0):+d}")
        else:
            code = reply.get("error")
            if code in ("interrupted", "rate_limited", None):
                return
            key = f"fail:{tool}:{code}:{args.get('recipe') or args.get('item') or ''}"
            self._add(key, f"{tool} {json.dumps(args)} failed: {code} ({reply.get('detail')})")

    def _add(self, key, text):
        if self.items.get(key) != text:
            self.items[key] = text
            self.path.write_text(json.dumps(self.items, indent=1))

    def text(self, limit=25) -> str:
        return "\n".join(f"- {v}" for v in list(self.items.values())[-limit:]) or "- (nothing learned yet)"


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
            "Decide your next action.",
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
               "hp": me["hp"], "hunger": me["hunger"], "energy": me["energy"], "thought": thought, "tool": tool,
               "args": args, "ok": ok, "error": reply.get("error"), "result": line, **meta}
        if source != "model" or not ok:
            rec["raw"] = raw[:500]
        self.decision_log.write(json.dumps(rec) + "\n")
        self.decision_log.flush()
        log.info("%s [%s %.1fs] %s | %s", self.name, source, meta.get("latency_s", 0), thought[:100], line[:140])

    def urgent(self, obs):
        """Reflexes that should not wait for a slow model: eat when starving, flee glooms."""
        me = obs["self"]
        inv = me["inventory"]
        if me["hunger"] <= 15:
            food = max((f for f in FOOD if inv.get(f)), key=lambda f: FOOD[f], default=None)
            if food:
                return "eat", {"item": food}, f"(reflex) starving, eating {food}"
        for e in obs["entities"]:
            if e["kind"] == "gloom" and abs(e["dx"]) + abs(e["dy"]) <= 2 and me["hp"] < 40:
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
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--data", default="agents")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(Thinker(a.name, a.persona, game_url=a.game, model_url=a.model_url,
                        data_dir=Path(a.data)).run(a.minutes))


if __name__ == "__main__":
    main()
