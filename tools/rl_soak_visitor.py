"""A scripted visitor for an RL player's soak on a QA copy (a --dev server, never the live one): at set minutes it
teleports beside the player (its x, y and height, from the spectator socket), hits it three times with bare hands,
then stands there 12 s and tells what the player did: hit back (the visitor's hp went down), moved off (how far),
or neither. A test harness, not a player.

    PYTHONPATH=src python tools/rl_soak_visitor.py --port 7792 --target Rook
"""

import argparse
import asyncio
import json
import math
import time

from websockets.asyncio.client import connect

from nexel_worlds.client import Client

AT = (2, 5, 8, 11, 14, 17, 20, 24)        # minutes: eight visits in a 30-minute soak


async def where(port, name):
    async with connect(f"ws://127.0.0.1:{port}/spectate", max_size=2 ** 24) as spec:      # (a fresh look each time)
        await spec.send(json.dumps({"op": "agents", "id": 1}))
        while True:
            m = json.loads(await spec.recv())
            if m.get("type") == "reply":
                return {a["name"]: a for a in m["agents"]}.get(name), {a["name"]: a for a in m["agents"]}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7792)
    p.add_argument("--target", default="Rook")
    p.add_argument("--name", default="Mira")
    a = p.parse_args()
    t0 = time.monotonic()
    async with Client(f"ws://127.0.0.1:{a.port}/agent") as g:
        await g.hello(a.name, f"qa-{a.name}")
        for at in AT:
            await asyncio.sleep(max(0.0, t0 + at * 60 - time.monotonic()))
            them, _ = await where(a.port, a.target)
            if not them:
                print(f"[{at} min] {a.target} not found", flush=True)
                continue
            await g.request("dev_teleport", x=them["x"] + 1, y=them["y"], z=them["z"])
            obs = await g.observe()
            tid = next((e["id"] for e in obs["entities"] if e.get("name") == a.target), None)
            hits = []
            for _ in range(3):
                r = await g.act("attack", target=tid) if tid else {"ok": False, "error": "not in view"}
                hits.append("ok" if r.get("ok") else r.get("error"))
                await asyncio.sleep(0.5)
            hp0, pos0 = obs["self"]["hp"], (them["x"], them["y"])
            await asyncio.sleep(12)
            them2, everyone = await where(a.port, a.target)
            me = everyone.get(a.name, {})
            moved = math.dist(pos0, (them2["x"], them2["y"])) if them2 else None
            print(f"[{at} min] {a.name} hits {a.target} at {pos0} z{them['z']} hp {them['hp']}: {hits}; 12 s later: "
                  f"visitor hp {hp0} -> {me.get('hp')} ({'hit back' if me.get('hp', hp0) < hp0 else 'not hit'}), "
                  f"{a.target} moved {moved:.1f} tiles, hp {them2['hp'] if them2 else '?'}", flush=True)
            await g.request("dev_teleport", x=them["x"] + 40, y=them["y"] + 40)     # (gone again)


if __name__ == "__main__":
    asyncio.run(main())
