"""A learned player on the real server: the policy trained in the sim (rl/npolicy.py, numpy) acts on the agent API's
observation with the same encoding (rl/encode.py) and the same timing as in training: a decision when the last action
ends (0.2 s at the soonest), when hurt, or every 0.5 s. No model in the action loop; with --model-url, someone
talking to it gets a line back from the local model (chat.py's prompt), and nothing else.
    python -m megastructure.rl.live --name Rook --policy rl_runs/r1/policy.npz --game ws://127.0.0.1:7790/agent"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np

from ..chat import chat_messages
from ..client import Client
from ..fastprompt import chat_parse
from .achieve import HOSTILE
from .encode import ACTIONS, action_name, encode, legal, to_command

MIN_S, MAX_S = 0.2, 0.5


class LiveBody:
    def __init__(self, a):
        from .npolicy import NumpyPolicy
        self.a, self.pol, self.rng = a, NumpyPolicy(a.policy), np.random.default_rng(a.seed)
        self.facing, self.last, self.last_ok, self.hurt = "S", 0, 0.0, 0.0
        self.stats = {"decisions": 0, "acts": 0, "fails": 0, "interrupted": 0, "idle": 0, "crafted": 0, "placed": 0,
                      "deaths": 0, "hits": 0, "won": 0, "lost": 0, "kills": 0, "chat": 0}
        self.crafts, self.reactions, self.foes, self.fails_by = set(), {}, {}, {}
        self.log = open(a.log, "a") if a.log else None

    def done(self, tool, args, kind, t):
        r = t.result()
        if not r.get("ok") and r.get("error") == "interrupted":
            self.stats["interrupted"] += 1
            return
        self.stats["acts"] += 1
        if not r.get("ok"):
            self.stats["fails"] += 1
            self.fails_by[r.get("error")] = self.fails_by.get(r.get("error"), 0) + 1
        elif tool == "craft":
            self.stats["crafted"] += 1
            self.crafts.add(args["recipe"])
        elif tool == "place":
            self.stats["placed"] += 1
        elif tool == "attack" and (r.get("result") or {}).get("killed"):
            self.stats["kills"] += 1
            if args.get("target") in self.foes:
                self.stats["won"] += 1
                self.foes.pop(args["target"])
        self.last_ok = 1.0 if r.get("ok") else -1.0

    async def say_back(self, game, http, speaker, text, me):
        msgs = chat_messages(self.a.name, self.a.persona, "", speaker, text, me=me)
        try:
            r = await http.post(self.a.model_url, timeout=30, json={"model": "default_model", "messages": msgs,
                                                                    "max_tokens": 80, "temperature": 0.8})
            line, _ = chat_parse(r.json()["choices"][0]["message"]["content"] or "")
        except Exception:
            return
        if line:
            await game.act("say", text=line[:120])
            self.stats["chat"] += 1

    async def play(self):
        http = None
        if self.a.model_url:
            import httpx
            http = httpx.AsyncClient()
        t_end = time.monotonic() + self.a.minutes * 60 if self.a.minutes else math.inf
        async with Client(self.a.game) as game:
            await game.hello(self.a.name, f"rl-{self.a.name}", self.a.persona)
            obs = await game.observe()
            hp, deaths, task, running, t_report = obs["self"]["hp"], obs["self"]["deaths"], None, None, time.monotonic()
            while time.monotonic() < t_end:
                me = obs["self"]
                for e in obs.get("events", []):
                    if http and e.get("type") in ("say", "shout") and e.get("from") != self.a.name:
                        asyncio.create_task(self.say_back(game, http, e.get("from"), str(e.get("text", ""))[:200], me))
                self.hurt = max(0.0, hp - me["hp"]) if me["deaths"] == deaths else hp
                if me["deaths"] > deaths:
                    self.stats["deaths"] += me["deaths"] - deaths
                    self.stats["lost"] += len(self.foes)
                    self.foes.clear()
                if self.hurt > 0:
                    self.stats["hits"] += 1
                    foe = next((b for b in obs.get("entities", []) if b.get("kind") in HOSTILE and math.hypot(b["dx"], b["dy"]) <= 5), None)
                    if foe:
                        self.foes[foe["id"]] = foe["kind"]
                hp, deaths = me["hp"], me["deaths"]
                i = self.pol.act(encode(obs, self.facing, self.last, self.last_ok, self.hurt), self.rng, legal(obs) if self.a.mask else None)
                kind = ACTIONS[i][0]
                if self.hurt > 0:
                    self.reactions[kind] = self.reactions.get(kind, 0) + 1
                if kind in ("step", "jump", "dig") and ACTIONS[i][1]:
                    self.facing = ACTIONS[i][1]
                self.stats["decisions"] += 1
                cmd = to_command(i, obs, self.facing)
                t0, self.last_ok = time.monotonic(), 0.0        # (as in the env: how the actions that end in this wait went)
                if cmd is not None and running and running[0] == cmd and not running[1].done():
                    cmd = None                                  # the same action again, still under way: it goes on
                if cmd is None:
                    self.stats["idle"] += me.get("action") is None
                else:
                    tool, args = cmd
                    task = asyncio.create_task(game.act(tool, **args))
                    task.add_done_callback(lambda t, tool=tool, args=args: self.done(tool, args, "", t))
                    running = (cmd, task)
                await asyncio.sleep(MIN_S)
                if task is not None and not task.done():
                    await asyncio.wait([task], timeout=MAX_S - (time.monotonic() - t0))
                self.last = i
                obs = await game.observe()
                if self.log and (self.stats["decisions"] % 200 == 0 or time.monotonic() - t_report > 60):
                    t_report = time.monotonic()
                    self.log.write(json.dumps(self.report(me, action_name(i))) + "\n")
                    self.log.flush()
        if http:
            await http.aclose()
        return self.report(obs["self"], "")

    def report(self, me, last):
        s = dict(self.stats)
        s["distinct_crafts"] = sorted(self.crafts)
        s["fail_pct"] = round(100 * s["fails"] / max(1, s["acts"]), 1)
        s["idle_pct"] = round(100 * s["idle"] / max(1, s["decisions"]), 1)
        return {"t": time.time(), **s, "reactions": self.reactions, "fails_by": self.fails_by,
                "inventory": me.get("inventory"), "hp": me.get("hp"), "last": last}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Rook")
    p.add_argument("--persona", default="a quiet player who learned the world by living in it")
    p.add_argument("--policy", default=str(Path(__file__).resolve().parents[3] / "rl_runs/policy.npz"))
    p.add_argument("--game", default="ws://127.0.0.1:7777/agent")
    p.add_argument("--model-url", default="", help="a local model for chat replies only (none: it does not talk)")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--log", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask", type=int, default=1, help="the policy was trained with the legal-action mask")
    a = p.parse_args()
    print(json.dumps(asyncio.run(LiveBody(a).play())))


if __name__ == "__main__":
    main()
