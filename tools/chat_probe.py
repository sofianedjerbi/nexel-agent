"""Talk to an AI player and time its answers: joins as another player (a --dev server: it teleports next to the
target, found from the target's decisions.jsonl), says a line every --every seconds, prints each answer's wait.

    PYTHONPATH=src python tools/chat_probe.py --game ws://127.0.0.1:7843/agent --target rt --log agents/rt/decisions.jsonl
"""

import argparse
import asyncio
import json
import statistics
import time

from megastructure.client import Client

LINES = ["hi! who are you?", "got any wood to spare?", "watch out, there's a boar around", "want to trade berries for stone?",
         "what are you building?", "can you help me find ore?", "nice weather today", "where do you live?"]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ws://127.0.0.1:7843/agent")
    p.add_argument("--target", required=True)
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--every", type=float, default=40)
    p.add_argument("--log", required=True, help="the target's decisions.jsonl (for where it stands)")
    a = p.parse_args()
    waits = []
    async with Client(a.game) as g:
        await g.hello("Ana", "probe-ana")
        end, i = time.monotonic() + a.minutes * 60, 0
        while time.monotonic() < end:
            await asyncio.sleep(a.every)
            pos = None
            for line in open(a.log).read().splitlines()[::-1]:
                rec = json.loads(line)
                if "pos" in rec:
                    pos = rec["pos"]
                    break
            if pos:
                await g.request("dev_teleport", x=pos[0] + 2, y=pos[1])
            await g.observe()                    # (drop what was heard before)
            text = LINES[i % len(LINES)]
            i += 1
            await g.act("say", text=text)
            t0 = time.monotonic()
            got = None
            while time.monotonic() - t0 < 15 and got is None:
                await asyncio.sleep(0.1)
                obs = await g.observe()
                got = next((e for e in obs["events"] if e.get("type") == "say" and e.get("from") == a.target), None)
            dt = time.monotonic() - t0
            if got:
                waits.append(dt)
            print(f"Ana: {text!r} -> {a.target}: {got and got['text']!r} after {dt:.1f}s", flush=True)
    if waits:
        print(f"answers {len(waits)}, p50 {statistics.median(waits):.1f}s, max {max(waits):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
