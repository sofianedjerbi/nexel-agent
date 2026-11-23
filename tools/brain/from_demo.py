"""Today's demo logs as a brain feed (docs/brain_feed.md), for the brain view's replay.

    python3 tools/brain/from_demo.py RUN_DIR [-o tools/brain/demo_replay.jsonl] [--minutes 13] [--name Rook]

RUN_DIR holds what a live run (rl/live.py --log, rl/mind.py) wrote: rook.jsonl (each decision: t, pos, hp, act, hurt;
a report each minute), mind.jsonl (each thought: the facts and the reply), memory.json, latency.jsonl (the voice).
Only what the logs say goes in, as well as they allow:
  - frames, one a second, from the decisions: pos, hp, act, facing (the way the last step, dig, jump or climb went);
    food, what it carries, its goal and the people in sight come from the last thought's facts (people: the distance
    only, no dx/dy), the game clock between two thoughts is read off the two;
  - no tiles, no policy probabilities, no legal count, no value: the logs do not have them (the page hides those);
  - thoughts as logged; chat: what people said (the mind's notes, to the second), the mind's lines and the voice's;
  - events: hits (the hit itself, from the mind's notes; or a loss of health), goal changes, asks, each minute's report;
  - memory: the notes on people replayed from the replies as rl/mind.py applies them (ends as memory.json).
The busiest stretch (--minutes long: hits, chat, goal changes, people about) is kept. Plain Python, no packages."""

import argparse
import json
import re
import time
from pathlib import Path

DIRS = ("N", "E", "S", "W")
FACING_KINDS = ("step", "jump", "dig", "dig_at", "climb")
NOT_NAMES = ("no_one", "nobody", "none", "someone", "person", "a name you see")
PERSON = re.compile(r"^(\S+): (\d+) tiles away, health ([\d.]+), holding (\S+)\. Your notes: (.*?)( \(your ally\))?\. "
                    r"Lately: (.*)$")


def read(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def clock_epoch(hms: str, near: float) -> float:
    """HH:MM:SS (the log machine's local time) as Unix time, on the day of `near`."""
    lt = time.localtime(near)
    h, m, s = (int(x) for x in hms.split(":"))
    t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, s, 0, 0, -1))
    return t - 86400 if t > near + 3600 else t


def game_minutes(text: str) -> int | None:
    m = re.search(r"day (\d+), (\d+):(\d+)", text)
    return int(m.group(1)) * 1440 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def game_clock(mins: float) -> str:
    d, r = divmod(int(mins), 1440)
    h, m = divmod(r, 60)
    return f"day {d}, {h:02d}:{m:02d}" + (" (night)" if h >= 18 or h < 5 else "")


def parse_facts(text: str) -> dict:
    """What a thought's facts say: hp, food, carrying, goal, people (and their notes' dated lines), asks."""
    out = {"people": [], "asks": []}
    for line in text.splitlines():
        if m := re.match(r"You: health ([\d.]+)/20, food ([\d.]+)/20, carrying (.*?)\. (day .*?)\.?$", line):
            out["hp"], out["food"] = float(m.group(1)), float(m.group(2))
            inv = {}
            for part in m.group(3).split(", "):
                k, _, n = part.rpartition(" ")
                if k and n.isdigit():
                    inv[k] = int(n)
            out["inv"], out["clock"] = inv, game_minutes(m.group(4))
        elif m := re.match(r"Your goal now: (\S+) ?(\S+)?(?: \(with (\S+)\))?, set (\d+) s ago\.", line):
            out["goal"] = {"kind": m.group(1), **({"item": m.group(2)} if m.group(2) else {}),
                           **({"with": m.group(3)} if m.group(3) else {}), "set_s": int(m.group(4))}
        elif m := re.match(r"(\S+) asked you for (\S+);", line):
            out["asks"].append((m.group(1), m.group(2)))
        elif m := PERSON.match(line):
            lately = [x for x in re.split(r"; (?=\d\d:\d\d:\d\d )", m.group(7).rstrip(".")) if x and x != "nothing"]
            out["people"].append({"name": m.group(1), "d": int(m.group(2)), "hp": float(m.group(3)),
                                  "ally": bool(m.group(6)), "lately": lately})
    return out


def num(v):
    return int(v) if float(v).is_integer() else v


def reply_of(raw: str) -> dict:
    try:
        return json.loads(re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", raw, flags=re.S), re.S).group(0))
    except (AttributeError, ValueError):
        return {"raw": raw[:300]}


def convert(run: Path, name: str) -> list:
    rows = read(run / f"{name.lower()}.jsonl")
    decisions = [r for r in rows if "act" in r]
    reports = [r for r in rows if "decisions" in r]
    thoughts = sorted(read(run / "mind.jsonl"), key=lambda d: d["t"])
    voice = read(run / "latency.jsonl")
    out = [{"k": "hello", "t": round(decisions[0]["t"] - 0.1, 1), "v": 1, "name": name}]

    # thoughts, chat, the notes' dated lines, goals, memory
    seen, relations, goal, goal_at = set(), {}, None, None
    parsed = []
    for d in thoughts:
        f = parse_facts(d["facts"])
        rep = reply_of(d.get("reply", ""))
        parsed.append((d["t"], f))
        line = {"k": "thought", "t": round(d["t"], 1), "facts": d["facts"], "reply": rep}
        if d.get("llm_s") is not None:
            line["took_s"] = d["llm_s"]
        out.append(line)
        if goal is None and f.get("goal"):
            goal, goal_at = {k: v for k, v in f["goal"].items() if k != "set_s"}, d["t"] - f["goal"]["set_s"]
        for p in f["people"]:
            for item in p["lately"]:
                hms, _, what = item.partition(" ")
                key = (p["name"], hms, what)
                if key in seen:
                    continue
                seen.add(key)
                t = round(clock_epoch(hms, d["t"]), 1)
                if m := re.match(r'said "(.*)"$', what):
                    out.append({"k": "chat", "t": t, "from": p["name"], "text": m.group(1)})
                elif m := re.match(r"hit you \(([\d.]+)\)", what):
                    out.append({"k": "event", "t": t, "what": "hurt", "text": f"{p['name']} hit {name} ({m.group(1)})", "n": num(m.group(1))})
                elif m := re.match(r"hit (\S+)", what):
                    out.append({"k": "event", "t": t, "what": "attack", "text": f"{p['name']} hit {m.group(1).split('#')[0]}"})
                else:
                    out.append({"k": "event", "t": t, "what": "give" if "gave" in what else "note", "text": f"{p['name']}: {what}"})
        for who, item in f["asks"]:
            if ("ask", who, item) not in seen:
                seen.add(("ask", who, item))
                out.append({"k": "event", "t": round(d["t"], 1), "what": "ask", "text": f"{who} asked {name} for {item}"})
        if "raw" in rep:
            continue
        g = {"kind": str(rep.get("goal") or "free").split()[0].lower()}
        if rep.get("item"):
            g["item"] = str(rep["item"]).strip().lower().replace(" ", "_")
        if rep.get("with"):
            g["with"] = str(rep["with"]).strip()
        if g["kind"] == "free":
            g = None
        if g != goal:
            goal, goal_at = g, d["t"]
            text = "no goal" if g is None else "goal: " + " ".join(x for x in (g["kind"], g.get("item"), g.get("with") and f"(with {g['with']})") if x)
            out.append({"k": "event", "t": round(d["t"] + 0.1, 1), "what": "goal", "text": text})
        parsed[-1] = (d["t"], dict(f, goal_after=goal, goal_at=goal_at))
        if rep.get("say"):
            out.append({"k": "chat", "t": round(d["t"] + 0.2, 1), "from": name, "text": str(rep["say"])[:120], "said": True})
        before = json.dumps(relations, sort_keys=True)
        for who, upd in (rep.get("relation_update") or {}).items():
            if isinstance(upd, dict) and who and who.lower() not in NOT_NAMES:
                r = relations.setdefault(who, {})
                if "ally" in upd:
                    r["ally"] = bool(upd["ally"])
                if upd.get("note"):
                    r["notes"] = str(upd["note"])[:200]
        if json.dumps(relations, sort_keys=True) != before:
            out.append({"k": "memory", "t": round(d["t"] + 0.3, 1), "people": json.loads(json.dumps(relations))})
    for v in voice:
        if v.get("said"):
            out.append({"k": "chat", "t": round(v["t"], 1), "from": name, "text": " ".join(v["said"].split())[:160], "said": True})

    # each minute's report, as one line of what it did
    prev = None
    for r in reports:
        if prev is not None and r["decisions"] >= prev["decisions"]:
            dd = {k: r.get(k, 0) - prev.get(k, 0) for k in ("crafted", "placed", "kills", "won", "deaths")}
            parts = [f"{v} {w}" for w, v in (("made", dd["crafted"]), ("placed", dd["placed"]), ("kills", dd["kills"]),
                                             ("fights won", dd["won"]), ("deaths", dd["deaths"])) if v > 0]
            if parts:
                out.append({"k": "event", "t": round(r["t"], 1), "what": "minute", "text": "last minute: " + ", ".join(parts)})
        prev = r

    # frames, one a second, from the decisions
    hits = sorted(x["t"] for x in out if x.get("what") == "hurt")
    facing, last_t, ti = "S", -1e9, -1
    fights = False
    for d in decisions:
        kind, _, rest = d["act"].partition("_")
        if kind == "dig" and rest.startswith("at"):
            kind, rest = "dig_at", rest[3:]
        if kind in FACING_KINDS and rest[:1] in DIRS:
            facing = rest[:1]
        if d.get("hurt", 0) > 0 and not any(abs(h - d["t"]) < 3 for h in hits):
            out.append({"k": "event", "t": round(d["t"], 1), "what": "hurt", "text": f"{name} lost {num(d['hurt'])} health", "n": num(d["hurt"])})
        if kind == "attack" and not fights:
            out.append({"k": "event", "t": round(d["t"], 1), "what": "attack", "text": f"{name} fights"})
        fights = kind == "attack"
        if d["t"] - last_t < 1.0:
            continue
        last_t = d["t"]
        while ti + 1 < len(parsed) and parsed[ti + 1][0] <= d["t"]:
            ti += 1
        fr = {"k": "frame", "t": round(d["t"], 1), "pos": d["pos"], "hp": num(d["hp"]), "facing": facing, "act": d["act"]}
        if ti >= 0:
            t0, f = parsed[ti]
            if "food" in f:
                fr["food"] = num(f["food"])
            if ti + 1 < len(parsed) and f.get("clock") is not None and parsed[ti + 1][1].get("clock") is not None:
                t1, c0, c1 = parsed[ti + 1][0], f["clock"], parsed[ti + 1][1]["clock"]
                fr["time"] = game_clock(c0 + (c1 - c0) * (d["t"] - t0) / max(1e-6, t1 - t0))
            fr["inv"] = f.get("inv", {})
            g = f.get("goal_after")
            if g:
                fr["goal"] = dict(g, since_s=round(d["t"] - f["goal_at"]), **({"have": fr["inv"].get(g["item"], 0)} if g.get("item") else {}))
            if d["t"] - t0 < 12:
                fr["beings"] = [{"name": p["name"], "kind": "agent", "d": p["d"], "hp": num(p["hp"]), **({"side": "ally"} if p["ally"] else {})}
                                for p in f["people"]]
            else:
                fr["beings"] = []
        out.append(fr)
    out.sort(key=lambda x: (x["t"], x["k"] != "hello"))
    return out


def busiest(lines: list, minutes: float) -> list:
    """The stretch of `minutes` with the most going on: hits, chat, goal changes, asks; people in sight."""
    w = {"chat": 3, "event": 2}
    t0s = [x["t"] for x in lines if x["k"] == "frame"]
    if not t0s or t0s[-1] - t0s[0] <= minutes * 60:
        return lines
    score = lambda a, b: sum(w.get(x["k"], 0) + (1 if x["k"] == "frame" and x.get("beings") else 0)
                             for x in lines if a <= x["t"] < b)
    start = max(t0s[:: 10], key=lambda a: score(a - 5, a + minutes * 60))
    a, b = start - 5, start + minutes * 60
    keep = [x for x in lines if a <= x["t"] < b and x["k"] != "hello"]
    mem = [x for x in lines if x["k"] == "memory" and x["t"] < a]
    head = [dict(lines[0], t=round(a, 1))] + ([dict(mem[-1], t=round(a, 1))] if mem else [])
    return head + keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("-o", "--out", default=str(Path(__file__).with_name("demo_replay.jsonl")))
    p.add_argument("--minutes", type=float, default=13)
    p.add_argument("--name", default="Rook")
    a = p.parse_args()
    lines = busiest(convert(Path(a.run), a.name), a.minutes)
    Path(a.out).write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines))
    n = {}
    for x in lines:
        n[x["k"]] = n.get(x["k"], 0) + 1
    print(f"{a.out}: {len(lines)} lines, {Path(a.out).stat().st_size // 1024} KB, "
          f"{(lines[-1]['t'] - lines[0]['t']) / 60:.1f} min, {n}")


if __name__ == "__main__":
    main()
