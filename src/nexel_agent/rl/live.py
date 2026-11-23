"""A learned player on the real server: the policy trained in the sim (rl/npolicy.py, numpy) acts on the agent API's
observation with the same encoding (rl/encode.py) and the same timing as in training: a decision when the last action
ends (0.2 s at the soonest), at once when hurt or someone comes into view, or after 5 s. No model in the action loop:
the policy chooses when to speak (say) and what to hand over (give); with --model-url the local model writes the words
(rl/voice.py), else a plain short line. The model to serve there: voice.MODEL (Qwen3-30B-A3B-Instruct-2507, 4-bit), or
voice.FALLBACK_MODEL (the 4B) on a Mac with less than ~20 GB free.
    python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080
    nexel-agent --name Rook          # the shipped models/rook.npz, the game at :7777, words from :8080
(`python -m nexel_agent` and `python -m nexel_agent.rl.live` are the same command.)"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np

from megastructure.client import Client
from .achieve import HOSTILE
from .brain import Brain
from .core import split
from .encode import ACTIONS, ITEMS, N_NEAR, action_name, encode, legal, to_command
from .social import SocialMemory
from .mind import goal_text
from .voice import Now, Voice

POLICY = Path(__file__).resolve().parents[3] / "models/rook.npz"     # (the shipped body: models/README.md)
MODEL_URL = "http://127.0.0.1:8080/v1/chat/completions"             # (the shared local model server, as players.py starts it)

MIN_S, IDLE_S, WAIT_MAX_S, POLL_S = 0.2, 0.5, 5.0, 0.25


class LiveBody:
    def __init__(self, a):
        from .npolicy import NumpyPolicy
        self.a, self.pol, self.rng = a, NumpyPolicy(a.policy), np.random.default_rng(a.seed)
        self.facing, self.last, self.last_ok, self.hurt = "S", 0, 0.0, 0.0
        self.stats = {"decisions": 0, "acts": 0, "fails": 0, "interrupted": 0, "idle": 0, "crafted": 0, "placed": 0,
                      "deaths": 0, "hits": 0, "won": 0, "lost": 0, "kills": 0, "chat": 0}
        self.crafts, self.reactions, self.foes, self.fails_by = set(), {}, {}, {}
        self.social, self.social_done, self.http, self.game = SocialMemory(a.name, ITEMS), set(), None, None
        self.stats.update(said=0, gave=0)
        self.tick, self.me, self.obs_entities, self.fighting = 0, {}, [], False
        self.mind = self.voice = None                       # (the LLM planner that sets goals: rl/mind.py; words: voice.py)
        self.now = Now(a.name, ITEMS)                            # (what is happening to it, for every prompt: rl/voice.py)
        self.log = open(a.log, "a") if a.log else None
        self.brain = Brain(a.brain, a.log and Path(a.log).with_name("brain.jsonl"), a.name, a.persona) if getattr(a, "brain", 0) else None

    def done(self, tool, args, r):
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
        elif tool in ("say", "give"):
            self.stats["said" if tool == "say" else "gave"] += 1
        elif tool == "attack":
            self.stats["swings"] = self.stats.get("swings", 0) + 1
            if (r.get("result") or {}).get("killed"):
                self.stats["kills"] += 1
                if args.get("target") in self.foes:
                    self.stats["won"] += 1
                    self.foes.pop(args["target"])
        near = min((math.hypot(b["dx"], b["dy"]) for b in self.obs_entities if b.get("kind") == "agent"), default=99)
        self.social_done |= set(self.social.achieved(tool, args, bool(r.get("ok")), self.tick, near,
                                                     (r.get("result") or {}).get("to")))
        self.last_ok = 1.0 if r.get("ok") else -1.0

    async def call(self, game, cmd):
        """One action as in the env (core.split): a tool call, or a climb's digs then jump."""
        r = None
        self.fighting = cmd[0] == "attack"
        for step in split(cmd):
            if self.fighting and r is not None and not r.get("ok"):
                break                                  # (the fight is over: the target fell or got away)
            if step[0] == "say":                       # the words: the LLM's (rl/voice.py), or the plain line
                step = ("say", {"text": await self.words(step[1].get("ask"), step[1]["text"])})
                if not step[1]["text"]:
                    continue                           # (it chose to say nothing)
            r = await game.act(step[0], **step[1])
            self.done(step[0], step[1], r)
        self.fighting = False
        return r

    async def wait(self, game, task, me, t0):
        """As in the env (rl/core.py): till the action ends (0.2 s at the soonest; idle, 0.5 s), or at once when hurt
        or someone comes into view, or 5 s at the latest. The events seen meanwhile are all kept. (Decision timing only:
        a hit or a newcomer never chooses what to do; the policy does.)"""
        seen = {b["id"] for b in self.obs_entities}
        events = []
        await asyncio.sleep(MIN_S)
        while True:
            obs = await game.observe()
            events += obs.get("events", [])
            self.listen(game, obs)
            s, dt = obs["self"], time.monotonic() - t0
            new = {b["id"] for b in obs.get("entities", [])} - seen
            hurt = s["hp"] < me["hp"] and not (self.fighting and task is not None and not task.done())
            if hurt or s["deaths"] != me["deaths"] or new or dt >= WAIT_MAX_S \
                    or (task.done() if task is not None else dt >= IDLE_S):
                obs["events"] = events
                self.obs_entities = obs.get("entities", [])
                return obs
            await asyncio.sleep(POLL_S)

    def listen(self, game, obs):
        """Each observation: into what is happening now; someone spoke: a salient event, straight to the voice
        (rl/voice.py: a burst of lines gets one answer), whose reply, if any, is said."""
        self.now.see(obs)
        for e in obs.get("events", []):
            if self.voice and e.get("type") in ("say", "shout") and e.get("from") != self.a.name:
                self.voice.hear(e.get("from"), str(e.get("text", ""))[:200], self.say, self.doing)

    async def say(self, line):
        await self.game.act("say", text=line)
        self.social.my_last = self.tick
        self.stats["chat"] += 1

    def doing(self, ask=None) -> str:
        return f"some {ask}" if ask else goal_text(self.mind.goal) if self.mind else ""

    async def words(self, ask=None, plain="hey") -> str:
        """The policy chose to speak (or to ask for something); the words are the LLM's, from the same facts - or none
        (it may stay silent; an answer under way is its turn already)."""
        if not self.voice:
            return plain if ask else "hey"
        if ask:
            self.now.ask(self.a.name, None, ask)
        if self.voice.answering:
            return ""
        self.stats["chat"] += 1
        return await self.voice.line([], self.doing(ask), kind="start")

    async def play(self):
        if self.a.model_url:
            import httpx
            from .mind import Mind
            self.http = httpx.AsyncClient()
            self.mind = Mind(self.a.name, self.a.persona, self.a.model_url, self.http,
                             Path(self.a.memory or f"agents/{self.a.name}/memory.json"), self.now)
            self.voice = self.mind.voice = Voice(self.a.name, self.a.persona, self.a.model_url, self.http,
                                                 self.mind.path.with_name("latency.jsonl"), self.now)
        t_end = time.monotonic() + self.a.minutes * 60 if self.a.minutes else math.inf
        async with Client(self.a.game) as game:
            self.game = game
            await game.hello(self.a.name, f"rl-{self.a.name}", self.a.persona)
            if self.a.teleport:                       # (a --dev server: put it somewhere first, for a demo or a test)
                x, y, z = (int(v) for v in self.a.teleport.split(","))
                await game.request("dev_teleport", x=x, y=y, z=z)
            obs = await game.observe()
            self.obs_entities = obs.get("entities", [])
            self.listen(game, obs)                    # (later ones: each poll of wait)
            hp, deaths, task, running, t_report = obs["self"]["hp"], obs["self"]["deaths"], None, None, time.monotonic()
            while time.monotonic() < t_end:
                me = self.me = obs["self"]
                self.tick = obs["tick"]
                self.social.see(obs)
                if self.mind:
                    self.mind.see(obs)
                    if self.mind.due():
                        ask = self.social.ask_open(obs["tick"])
                        self.mind.asks = (f"{self.social.asked_by} asked you for {ask}; you carry "
                                          f"{me.get('inventory', {}).get(ask, 0)} {ask}") if ask else ""
                        asyncio.create_task(self.mind.think(obs))
                    if self.mind.to_say and self.social.can_talk(obs["tick"]):    # (its own line: the same chat cooldown)
                        line, self.mind.to_say = self.mind.to_say, None
                        asyncio.create_task(game.act("say", text=line))
                        self.stats["said"] += 1
                self.hurt = max(0.0, hp - me["hp"]) if me["deaths"] == deaths else hp
                if me["deaths"] > deaths:
                    self.stats["deaths"] += me["deaths"] - deaths
                    self.stats["lost"] += len(self.foes)
                    self.foes.clear()
                if self.hurt > 0:
                    self.stats["hits"] += 1
                    foe = next((b for b in obs.get("entities", []) if b.get("kind") in HOSTILE + ("agent",) and math.hypot(b["dx"], b["dy"]) <= 5), None)
                    if foe:
                        self.foes[foe["id"]] = foe["kind"]
                hp, deaths = me["hp"], me["deaths"]
                goal = self.mind.goal if self.mind else None
                enc = encode(obs, self.facing, self.last, self.last_ok, self.hurt, self.social.flags(obs), goal,
                             self.social.people_flags(obs, N_NEAR))
                mask = legal(obs, self.facing, self.social, goal) if self.a.mask else None
                i = self.pol.act(enc, self.rng, mask)
                kind = ACTIONS[i][0]
                if self.hurt > 0:
                    self.reactions[kind] = self.reactions.get(kind, 0) + 1
                if kind in ("step", "jump", "dig", "dig_at", "climb") and ACTIONS[i][1]:
                    self.facing = ACTIONS[i][1]
                self.stats["decisions"] += 1
                self.now.did(action_name(i))
                if self.brain:
                    self.brain.see(obs, self, enc, mask, i)
                cmd = to_command(i, obs, self.facing, self.social, goal)
                t0, self.last_ok = time.monotonic(), 0.0        # (as in the env: how the actions that end in this wait went)
                if cmd is not None and running and running[0] == cmd and not running[1].done():
                    cmd = None                                  # the same action again, still under way: it goes on
                if cmd is None:
                    self.stats["idle"] += me.get("action") is None
                else:
                    if running and not running[1].done():
                        running[1].cancel()                     # (a new action replaces the one under way)
                    if cmd[0] == "say":                  # (the chat cooldown starts now, as in the sim, not when the line comes back)
                        self.social.my_last = obs["tick"]
                    task = asyncio.create_task(self.call(game, cmd))
                    running = (cmd, task)
                under_way = running[1] if running and not running[1].done() else None
                self.last = i
                obs = await self.wait(game, under_way, me, t0)
                if self.log:                 # (each decision: where, what, how hurt; a full report each minute)
                    s = obs["self"]
                    self.log.write(json.dumps({"t": round(time.time(), 2), "pos": [s["x"], s["y"]], "hp": s["hp"],
                                               "act": action_name(i), "hurt": self.hurt}) + "\n")
                    if time.monotonic() - t_report > 60:
                        t_report = time.monotonic()
                        self.log.write(json.dumps(self.report(s, action_name(i))) + "\n")
                    self.log.flush()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self.http:
            await self.http.aclose()
        return self.report(obs["self"], "")

    def report(self, me, last):
        s = dict(self.stats)
        s["distinct_crafts"] = sorted(self.crafts)
        s["social"] = sorted(self.social_done)
        if self.mind:
            s["mind"] = dict(self.mind.stats, goal=[self.mind.goal.kind, self.mind.goal.item, self.mind.goal.person],
                             allies=sorted(n for n, r in self.mind.relations.items() if r.get("ally")), aims=self.mind.aims)
        s["fail_pct"] = round(100 * s["fails"] / max(1, s["acts"]), 1)
        s["idle_pct"] = round(100 * s["idle"] / max(1, s["decisions"]), 1)
        return {"t": time.time(), **s, "reactions": self.reactions, "fails_by": self.fails_by,
                "inventory": me.get("inventory"), "hp": me.get("hp"), "last": last}


def main():
    p = argparse.ArgumentParser(prog="nexel-agent", description="The AI player of Nexel Worlds: a trained body, a local model for its plans and words.")
    p.add_argument("--name", default="Rook")
    p.add_argument("--persona", default="a quiet player who learned the world by living in it; wants a house of its own with "
                   "a bed and a door, and good tools, and gets what it lacks from other players by asking or trading")
    p.add_argument("--policy", default=str(POLICY), help="the learned body (default: the shipped models/rook.npz)")
    p.add_argument("--game", default="ws://127.0.0.1:7777/agent")
    p.add_argument("--model-url", default=MODEL_URL, help="a local model server for the words and the planner (none: it does not "
                   "talk; if nothing answers there it plays on in silence); serve voice.MODEL there (Qwen3-30B-A3B-Instruct-2507 "
                   "4-bit), or voice.FALLBACK_MODEL")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--log", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask", type=int, default=1, help="the policy was trained with the legal-action mask")
    p.add_argument("--teleport", default="", help="x,y,z: on a --dev server, start there")
    p.add_argument("--brain", type=int, default=0, help="serve what it sees and thinks on 127.0.0.1:PORT/feed (rl/brain.py)")
    p.add_argument("--memory", default="", help="where the LLM layer keeps its notes on people (memory.json)")
    a = p.parse_args()
    print(json.dumps(asyncio.run(LiveBody(a).play())))


if __name__ == "__main__":
    main()
