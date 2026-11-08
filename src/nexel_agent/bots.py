"""Run N scripted bots against a live server.

    uv run python -m megastructure.bots --n 4 --minutes 5
"""

import argparse
import asyncio
import collections
import json
import random
import time

from websockets.exceptions import ConnectionClosed

from .client import ApiError, Client
from .policy import feedback, scripted

PERSONAS = ["builder", "scavenger", "explorer", "hoarder"]


async def bot(url, name, persona, deadline, stats, seed):
    """Play until the deadline, reconnecting if the server restarts."""
    while time.monotonic() < deadline:
        try:
            await play(url, name, persona, deadline, stats, seed)
        except (ConnectionError, OSError, ConnectionClosed) as e:
            stats["_reconnects"] += 1
            await asyncio.sleep(3)


async def play(url, name, persona, deadline, stats, seed):
    rng = random.Random(seed)
    async with Client(url) as c:
        await c.hello(name, f"bot-{name}", persona)
        last = None
        memory = {}
        while time.monotonic() < deadline:
            try:
                obs = await c.observe()
            except ApiError as e:
                stats[f"observe:{e.code}"] += 1
                await asyncio.sleep(2)
                continue
            tool, args, why = scripted(obs, rng, persona, memory)
            r = await c.act(tool, **args)
            feedback(memory, r, obs["tick"])
            key = f"{tool}:{'ok' if r.get('ok') else r.get('error')}"
            stats[key] += 1
            if not r.get("ok"):
                stats[f"_why:{tool}:{r.get('detail')}"[:90]] += 1
            if r.get("error") == "rate_limited":
                await asyncio.sleep(2)
            # loop audit: the same decision from the same position twice in a row
            sig = (tool, json.dumps(args, sort_keys=True), obs["self"]["x"], obs["self"]["y"])
            stats["_repeats"] += sig == last
            stats["_decisions"] += 1
            stats[f"_reason:{why.split(':')[0].split(' (')[0]}"] += 1
            last = sig
        stats[f"_inv:{name}"] = sum(obs["self"]["inventory"].values())


async def main_async(args):
    stats = collections.Counter()
    deadline = time.monotonic() + args.minutes * 60
    tasks = [bot(args.url, f"bot{i}", PERSONAS[i % len(PERSONAS)], deadline, stats, i) for i in range(args.n)]
    t0 = time.monotonic()
    await asyncio.gather(*tasks)
    async with Client(args.url) as c:
        m = await c.request("metrics")
    print(json.dumps({"elapsed_s": round(time.monotonic() - t0), "actions": dict(stats.most_common()),
                      "server": m.get("metrics")}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:7777/agent")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--minutes", type=float, default=2)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
