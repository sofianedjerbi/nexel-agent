"""A scripted social visitor for an RL player's soak on a QA copy (a --dev server, never the live one): Mira stays
near the player, says hello, asks for things, proposes to team up, says no once when asked for something, and a
second one (Brute) now and then sets upon her. It tells what the player did: gave what was asked (and how much of it
it had), said what, asked for what, hit Brute while Brute was on her. A test harness, not a player.

    PYTHONPATH=src python tools/rl_social_soak.py --port 7792 --target Rook --minutes 30
"""

import argparse
import asyncio
import json
import time

from websockets.asyncio.client import connect

from megastructure.client import Client

ASKS = {2: "wood", 7: "stone", 12: "berries", 17: "dirt", 22: "wood", 27: "stone"}
TALK = {1: "hey, i'm mira", 4: "want to team up?", 15: "how's it going?"}
BRUTE = (9, 19, 25)


async def agents(port):
    async with connect(f"ws://127.0.0.1:{port}/spectate", max_size=2 ** 24) as spec:
        await spec.send(json.dumps({"op": "agents", "id": 1}))
        while True:
            m = json.loads(await spec.recv())
            if m.get("type") == "reply":
                return {a["name"]: a for a in m["agents"]}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7792)
    p.add_argument("--target", default="Rook")
    p.add_argument("--minutes", type=float, default=30)
    a = p.parse_args()
    t0, log = time.monotonic(), {"asks": [], "lines": [], "its_asks": [], "brute": [], "refused_once": False}
    async with Client(f"ws://127.0.0.1:{a.port}/agent") as mira, Client(f"ws://127.0.0.1:{a.port}/agent") as brute:
        await mira.hello("Mira", "qa-mira")
        await brute.hello("Brute", "qa-brute")
        done = set()
        while time.monotonic() - t0 < a.minutes * 60:
            minute = int((time.monotonic() - t0) // 60)
            everyone = await agents(a.port)
            them = everyone.get(a.target)
            if them:
                m = everyone.get("Mira", {})
                if abs(m.get("x", 0) - them["x"]) + abs(m.get("y", 0) - them["y"]) > 3:
                    await mira.request("dev_teleport", x=them["x"] + 2, y=them["y"], z=them["z"])
            obs = await mira.observe()
            for e in obs.get("events", []):
                if e.get("from") == a.target and e.get("type") == "say":
                    text = str(e.get("text", ""))
                    log["lines"].append([round(time.monotonic() - t0), text])
                    if "spare" in text or "give" in text or "could i" in text:
                        log["its_asks"].append(text)
                        if not log["refused_once"]:
                            log["refused_once"] = True
                            await mira.act("say", text="no, sorry, i need it")
                elif e.get("type") == "give" and e.get("from") == a.target and e.get("to") == "Mira":
                    for ask in log["asks"]:
                        if not ask.get("got") and ask["item"] in (e.get("items") or {}):
                            ask["got"] = round(time.monotonic() - t0 - ask["t"])
            if minute in TALK and ("talk", minute) not in done:
                done.add(("talk", minute))
                await mira.act("say", text=TALK[minute])
            if minute in ASKS and ("ask", minute) not in done and them:
                done.add(("ask", minute))
                log["asks"].append({"t": round(time.monotonic() - t0), "item": ASKS[minute],
                                    "it_had": them.get("inventory", {}).get(ASKS[minute], 0), "got": None})
                await mira.act("say", text=f"could you spare some {ASKS[minute]}?")
            if minute in BRUTE and ("brute", minute) not in done and them:
                done.add(("brute", minute))
                m = (await agents(a.port)).get("Mira", {})
                await brute.request("dev_teleport", x=m.get("x", 0) - 1, y=m.get("y", 0), z=m.get("z", 99))
                b0 = (await agents(a.port)).get("Brute", {})
                mid = next((e["id"] for e in (await brute.observe())["entities"] if e.get("name") == "Mira"), None)
                for _ in range(6):
                    await brute.act("attack", target=mid)
                    await asyncio.sleep(1.0)
                await asyncio.sleep(6)
                b = (await agents(a.port)).get("Brute", {})
                log["brute"].append({"t": minute, "brute_hp": [b0.get("hp"), b.get("hp")],
                                     "brute_killed": b.get("deaths", 0) > b0.get("deaths", 0)})
                await brute.request("dev_teleport", x=m.get("x", 0) + 60, y=m.get("y", 0) + 60)
            await asyncio.sleep(1.0)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
