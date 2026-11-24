"""A scripted talker for timing an RL player's replies on a QA copy (a --dev server, never the live one): Mira
stays beside the player and says a line every 20-40 s; each reply heard is timed from her line. A test harness.

    PYTHONPATH=src python tools/rl_talk_soak.py --port 7792 --target Rook --minutes 10
"""

import argparse
import asyncio
import json
import random
import time

from websockets.asyncio.client import connect

from nexel_worlds.client import Client

LINES = ["hey, i'm mira", "what are you up to?", "got any wood to spare?", "nice day for it", "want to team up?",
         "seen any wolves round here?", "where do you sleep at night?", "i found some stone up north",
         "can you help me build a wall?", "how's your food holding up?", "thanks, see you around"]


async def agents(port):
    async with connect(f"ws://127.0.0.1:{port}/spectate", max_size=2 ** 24) as spec:
        await spec.send(json.dumps({"op": "agents", "id": 1}))
        while True:
            m = json.loads(await spec.recv())
            if m.get("type") == "reply":
                return {x["name"]: x for x in m["agents"]}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7792)
    p.add_argument("--target", default="Rook")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--out", default="talk.json")
    a = p.parse_args()
    rng, t0, said, log = random.Random(7), time.monotonic(), [], []
    async with Client(f"ws://127.0.0.1:{a.port}/agent") as mira:
        await mira.hello("Mira", "qa-mira")
        next_at, k = time.monotonic() + 5, 0
        while time.monotonic() - t0 < a.minutes * 60:
            them = (await agents(a.port)).get(a.target)
            obs = await mira.observe()
            if them:
                m = obs.get("me", {})
                if abs(m.get("x", 0) - them["x"]) + abs(m.get("y", 0) - them["y"]) > 3:
                    await mira.request("dev_teleport", x=them["x"] + 2, y=them["y"], z=them["z"])
            for e in obs.get("events", []):
                if e.get("type") == "say" and e.get("from") == a.target:
                    now = time.monotonic()
                    last = said[-1] if said else None
                    log.append({"t": round(now - t0, 1), "text": e.get("text"),
                                "after_mira_s": round(now - last[0], 2) if last else None,
                                "to_line": last[1] if last else None})
            if time.monotonic() >= next_at and them:
                line = LINES[k % len(LINES)]
                await mira.act("say", text=line)
                said.append((time.monotonic(), line))
                k += 1
                next_at = time.monotonic() + rng.uniform(20, 40)
            await asyncio.sleep(0.25)
    json.dump({"mira_lines": len(said), "replies": log}, open(a.out, "w"), indent=1)
    print(f"mira said {len(said)}, heard {len(log)} lines")


if __name__ == "__main__":
    asyncio.run(main())
