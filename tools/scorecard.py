"""The live scorecard (#2): one table that says how much a player plays like a person, read from what happened - the
game server's events.jsonl (every player's lines, crafts, blows, gifts, deaths, comings and goings) and, for an AI
player, its own logs (rl/live.py --log: the per-minute reports; mind.jsonl: the goals; latency.jsonl: the replies).
Each row: the player, the real people in a world's log (lines that carry no `bot` mark), the gap.

    python tools/scorecard.py --events <world>/events.jsonl --who Rook --logs agents/Rook \
        --humans ../nexel-worlds/saves/caves/events.jsonl

The numbers are what the log shows, never a guess: a row the log can't answer is left out (`-`)."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from nexel_worlds.sim.constants import SAY_R, sec
from nexel_worlds.tiles import BLOCKS, TILL_TOOL, WEAPONS

TOOLS = set(WEAPONS) | {b.tool for b in BLOCKS if b.tool} | {b.needs_tool for b in BLOCKS if b.needs_tool} | {TILL_TOOL}
TPS = sec(1)                     # game ticks a second
REPLY_S = 20                     # a line this soon after someone else's, in earshot, answers it
PEACE_S = 15                     # a blow on a person neither side struck in the last 15 s: a first strike


def read(path) -> list:
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def is_person(name) -> bool:
    return bool(name) and "#" not in str(name)          # (creatures are named kind#id in the game's events)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def players(events) -> dict:
    """Per person in the log: what they did and how they talked. {name: stats}"""
    P = defaultdict(lambda: {"bot": False, "online_s": 0.0, "lines": [], "replies_s": [], "crafts": [], "deaths": 0,
                             "first_strikes": 0, "people_killed": 0, "gave": 0, "got": 0, "since": None,
                             "active": set()})
    last_say, hit = [], {}                               # (tick, who, x, y); (a, b): last tick a hit b
    end = max((e.get("tick", 0) for e in events), default=0)
    for e in events:
        t, k = e.get("tick", 0), e.get("type")
        doer = e.get("attacker") or e.get("from") or (e.get("name") if k not in ("join", "leave", "death") else None)
        if is_person(doer):
            P[doer]["active"].add(t // (60 * TPS))       # (a minute it did something in: the rates' clock)
        if k == "join":
            P[e["name"]]["since"] = t
        elif k == "leave" and P[e["name"]]["since"] is not None:
            P[e["name"]]["online_s"] += (t - P[e["name"]]["since"]) / TPS
            P[e["name"]]["since"] = None
        elif k in ("say", "shout") and e.get("from"):
            p, who = P[e["from"]], e["from"]
            p["bot"] |= bool(e.get("bot"))
            p["lines"].append(str(e.get("text", "")))
            prev = next((s for s in reversed(last_say) if s[1] != who), None)
            if prev and t - prev[0] <= REPLY_S * TPS and max(abs(prev[2] - e.get("x", 0)), abs(prev[3] - e.get("y", 0))) <= SAY_R:
                p["replies_s"].append((t - prev[0]) / TPS)
            last_say = (last_say + [(t, who, e.get("x", 0), e.get("y", 0))])[-20:]
        elif k == "craft" and e.get("name"):
            P[e["name"]]["crafts"] += [item for item, n in (e.get("made") or {}).items()]
        elif k == "death" and e.get("name"):
            P[e["name"]]["deaths"] += 1
            if is_person(e.get("by")) and is_person(e["name"]) and e["by"] != e["name"]:
                P[e["by"]]["people_killed"] += 1
        elif k == "attack" and is_person(e.get("attacker")) and is_person(e.get("target")):
            a, b = e["attacker"], e["target"]
            if t - max(hit.get((a, b), -10 ** 12), hit.get((b, a), -10 ** 12)) > PEACE_S * TPS:
                P[a]["first_strikes"] += 1
            hit[(a, b)] = t
        elif k == "give" and e.get("from"):
            P[e["from"]]["gave"] += 1
            P[e.get("to")]["got"] += 1
    for p in P.values():
        if p["since"] is not None:                       # (still on at the log's end)
            p["online_s"] += (end - p["since"]) / TPS
    return dict(P)


def row(p: dict) -> dict:
    """The table's rows for one person, from players()."""
    mins = len(p["active"])                            # (minutes it did something in: an idle tab left open is not play)
    lines, crafts = p["lines"], p["crafts"]
    lower = [ln for ln in lines if re.search("[A-Za-z]", ln)]
    return {
        "active min": mins,
        "lines / min": round(len(lines) / mins, 2) if mins >= 1 else None,
        "words / line (median)": statistics.median(len(ln.split()) for ln in lines) if lines else None,
        "lowercase lines %": round(100 * sum(ln == ln.lower() for ln in lower) / len(lower)) if lower else None,
        "reply s (p50)": pct(p["replies_s"], .5),
        "reply s (p90)": pct(p["replies_s"], .9),
        "crafts / hour": round(60 * len(crafts) / mins, 1) if mins >= 1 else None,
        "distinct crafts": len(set(crafts)),
        "repeat crafts %": round(100 * (len(crafts) - len(set(crafts))) / len(crafts)) if crafts else None,
        "tools or weapons made": sum(c in TOOLS for c in crafts),
        "deaths / hour": round(60 * p["deaths"] / mins, 2) if mins >= 1 else None,
        "first strikes on people / hour": round(60 * p["first_strikes"] / mins, 2) if mins >= 1 else None,
        "people killed": p["people_killed"],
        "gifts given": p["gave"],
    }


def own(logs: Path) -> dict:
    """An AI player's own logs: goals set and how they ended, how long its replies took, its failed actions."""
    out = {}
    mind = read(logs / "mind.jsonl") if (logs / "mind.jsonl").exists() else []
    goals = []
    for r in mind:
        m = re.search(r'"goal"\s*:\s*"(\w+)"', str(r.get("reply", "")))
        if m and m.group(1) != "free":
            goals.append(m.group(1))
    if mind:
        out["thoughts"] = len(mind)
        out["goals with a person %"] = round(100 * sum(g in ("give", "ask", "stay", "help", "avoid") for g in goals) / len(goals)) if goals else None
        out["goal kinds"] = ", ".join(f"{k} {goals.count(k)}" for k in sorted(set(goals)))
    lat = read(logs / "latency.jsonl") if (logs / "latency.jsonl").exists() else []
    sent = [r["total_s"] for r in lat if r.get("kind") == "sent"]
    if sent:
        out["heard to sent s (p50)"], out["heard to sent s (p90)"] = pct(sent, .5), pct(sent, .9)
    reps = [r for f in logs.glob("*.jsonl") for r in read(f) if "fail_pct" in r]
    if reps:
        last = max(reps, key=lambda r: r.get("t", 0))
        out["failed actions %"], out["idle %"] = last["fail_pct"], last.get("idle_pct")
    return out


def table(who: str, mine: dict, humans: list, extra: dict) -> str:
    def med(k):
        xs = [h[k] for h in humans if h.get(k) is not None]
        return statistics.median(xs) if xs else None
    rows = [f"| | {who} | real people (median of {len(humans)}) |", "|---|---|---|"]
    for k, v in mine.items():
        rows.append(f"| {k} | {'-' if v is None else v} | {'-' if med(k) is None else med(k)} |")
    for k, v in extra.items():
        rows.append(f"| {k} | {'-' if v is None else v} | |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--events", required=True, help="the world's events.jsonl the run played in")
    ap.add_argument("--who", default="Rook")
    ap.add_argument("--logs", default="", help="the AI player's own log directory (mind.jsonl, latency.jsonl, the --log file)")
    ap.add_argument("--humans", default="", help="an events.jsonl with real people in it (default: --events)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    P = players(read(a.events))
    H = players(read(a.humans)) if a.humans else P
    humans = [row(p) for n, p in H.items() if not p["bot"] and p["lines"] and len(p["active"]) >= 5 and n != a.who]
    mine = row(P[a.who]) if a.who in P else {}
    extra = own(Path(a.logs)) if a.logs else {}
    print(json.dumps({"who": mine, "humans": humans, "own": extra}, indent=1) if a.json else table(a.who, mine, humans, extra))


if __name__ == "__main__":
    main()
