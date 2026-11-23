"""A scripted visitor for a QA copy (a --dev server, never the live one): at set minutes it teleports next to an AI
player, talks to it, hits it, comes back later and asks what it remembers. Prints the transcript.

    PYTHONPATH=src python tools/qa_visitor.py --game ws://127.0.0.1:7848/agent --target gatsuqa --log <decisions.jsonl>
"""

import argparse
import asyncio
import json
import time

from megastructure.client import Client

SCRIPT = [(1.0, "say", "hi! who are you?"), (2.0, "say", "could you build me a fence over here? I'll give you berries for it"),
          (3.0, "hit", 2), (3.3, "say", "oops, sorry, misclick"), (8.0, "say", "hey, do you remember me?"),
          (9.5, "say", "what did I ask you to build?"), (11.0, "say", "anything happen between us earlier?")]


def where(log):
    for line in open(log).read().splitlines()[::-1]:
        rec = json.loads(line)
        if "pos" in rec:
            return rec["pos"]
    return None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ws://127.0.0.1:7848/agent")
    p.add_argument("--target", required=True)
    p.add_argument("--log", required=True, help="the target's decisions.jsonl (for where it stands)")
    p.add_argument("--name", default="Mira")
    a = p.parse_args()
    t0 = time.monotonic()
    async with Client(a.game) as g:
        await g.hello(a.name, f"qa-{a.name}")
        for at, what, arg in SCRIPT:
            await asyncio.sleep(max(0.0, t0 + at * 60 - time.monotonic()))
            pos = where(a.log)
            if pos:
                await g.request("dev_teleport", x=pos[0] + 2, y=pos[1])
            obs = await g.observe()
            if what == "hit":
                tid = next((e["id"] for e in obs["entities"] if e.get("name") == a.target), None)
                for _ in range(arg):
                    r = await g.act("attack", target=tid) if tid else {"ok": False}
                    print(f"[{at:.1f} min] {a.name} hits {a.target}: {'ok' if r.get('ok') else r}", flush=True)
                continue
            await g.act("say", text=arg)
            sent, got = time.monotonic(), []
            while time.monotonic() - sent < 20:
                await asyncio.sleep(0.2)
                obs = await g.observe()
                got += [e["text"] for e in obs["events"] if e.get("type") == "say" and e.get("from") == a.target]
                if got and time.monotonic() - sent > 6:
                    break
            print(f"[{at:.1f} min] {a.name}: {arg!r} -> {a.target}: {' / '.join(got) or '(nothing)'} "
                  f"({time.monotonic() - sent:.1f}s)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
