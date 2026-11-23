"""Read a thinker's decisions.jsonl and report a soak: actions by source and kind, distance travelled, inventory
gained, stalls the watchdog declared, and every 20 s window with < 3 tiles of net progress and nothing gained.

    python tools/soak_report.py agents/gatsu0/decisions.jsonl [saves/<world>/events.jsonl]

With the world's events.jsonl: the player's deaths and their causes, and who hit it. Chats and diary entries are
printed from the decisions log.
"""

import json
import re
import sys
from collections import Counter

sys.path.insert(0, "src")
from megastructure.ladder import Record  # noqa: E402


def main(path, events=None):
    recs = [json.loads(x) for x in open(path)]
    acts = [r for r in recs if "tool" in r]
    stalls = [r for r in recs if r.get("kind") == "stall"]
    calls = [r for r in recs if r.get("kind") == "call"]
    span = acts[-1]["t"] - acts[0]["t"]
    print(f"{len(acts)} actions in {span / 60:.1f} min ({len(acts) / span:.2f}/s), {len(calls)} model calls")
    print("by source:", dict(Counter(r["source"] for r in acts).most_common()))
    print("by kind:", dict(Counter(f"{r['tool']}{'' if r['ok'] else ' FAILED ' + str(r['error'])}" for r in acts).most_common(14)))
    path_len = sum(abs(b["pos"][0] - a["pos"][0]) + abs(b["pos"][1] - a["pos"][1]) for a, b in zip(acts, acts[1:]))
    a, b = acts[0]["pos"], acts[-1]["pos"]
    print(f"from {a} to {b}: net {abs(b[0] - a[0]) + abs(b[1] - a[1])} tiles, walked {path_len} tiles")
    got = Counter()
    for r in acts:
        if r["ok"]:
            for n, k in re.findall(r"([+-]\d+) ([a-z_]+)", r["result"].split("->", 1)[-1]):
                got[k] += int(n)
    print("inventory change:", dict(got))
    progress(acts)
    social(recs, acts, events)
    print(f"hp {acts[0]['hp']} -> {acts[-1]['hp']}, food {acts[0]['hunger']} -> {acts[-1]['hunger']}")
    print("stalls declared:", [(s["routine"], s["at"]) for s in stalls])
    bad, i = [], 0
    for j, r in enumerate(acts):                 # windows of 20 s: net < 3 tiles and the result lines gained nothing
        while acts[i]["t"] < r["t"] - 20:
            i += 1
        if r["t"] - acts[i]["t"] < 19 or (bad and acts[i]["t"] < bad[-1][1]):
            continue
        w = acts[i:j + 1]
        net = abs(r["pos"][0] - w[0]["pos"][0]) + abs(r["pos"][1] - w[0]["pos"][1])
        gain = any(re.search(r"[+-]\d+ [a-z]", x["result"].split("->", 1)[-1]) for x in w if x["ok"])
        if net < 3 and not gain:
            bad.append((round(w[0]["t"] - acts[0]["t"]), r["t"], tuple(r["pos"]), Counter(x["source"] for x in w).most_common(2)))
    print(f"20 s windows with < 3 tiles and no gain: {len(bad)}", [(b[0], b[2], b[3]) for b in bad[:12]])


def progress(acts):
    """Crafted, placed, kinds gained, failures by error, and the milestones replayed from the log (minute reached)."""
    ok = lambda tool: Counter(str(r["args"].get("recipe" if tool == "craft" else "item")) for r in acts if r["tool"] == tool and r["ok"])
    print("crafted:", dict(ok("craft")), "| placed:", dict(ok("place")))
    print("failures:", dict(Counter(f"{r['tool']} {r['error']}" for r in acts if not r["ok"]).most_common(12)))
    rec, inv, t0, kinds = Record(), Counter(), acts[0]["t"], set()
    for r in acts:
        if r["ok"]:
            for n, k in re.findall(r"([+-]\d+) ([a-z_]+)", r["result"].split("->", 1)[-1]):
                inv[k] += int(n)
                kinds |= {k} if int(n) > 0 else set()
        rec.seen(dict(inv), (r["t"] - t0) / 60)
        rec.did(r["tool"], r["args"], r["ok"], (r["t"] - t0) / 60)
    print(f"distinct kinds gained: {len(kinds)} {sorted(kinds)}")
    print("milestones (minute):", {m: round(v, 1) for m, v in rec.reached.items()})


def social(recs, acts, events):
    """Deaths (cause, minute), hits taken, every chat and diary entry."""
    t0, name = acts[0]["t"], acts[0]["agent"]
    if events:
        evs = [json.loads(x) for x in open(events) if '"death"' in x or '"attack"' in x]
        deaths = [e for e in evs if e["type"] == "death" and e.get("name") == name]
        hits = Counter(e.get("attacker") for e in evs if e["type"] == "attack" and e.get("target") == name)
        print(f"deaths: {len(deaths)}", [(e.get("cause"), e.get("by")) for e in deaths], "| hit by:", dict(hits))
    for r in recs:
        if r.get("kind") == "chat":
            print(f"  chat [{(r['t'] - t0) / 60:.1f} min] {r['to']}: {r['heard']!r} -> {r['said']!r}"
                  + (f" intent={r.get('intent') or r.get('do')}" if r.get("intent") or r.get("do") else "") + (f" promise={r['promise']!r}" if r.get("promise") else ""))
        elif r.get("kind") == "diary":
            print(f"  diary [{(r['t'] - t0) / 60:.1f} min] {r['text'][:300]!r}")
    for r in recs:
        if r.get("kind") == "cards":
            print(f"  cards written [{(r['t'] - t0) / 60:.1f} min, {r['source']}, {r['latency_s']}s]:",
                  [(c["when"], c["do"]) for c in r["new"]])
        elif r.get("kind") == "card":
            print(f"  card fired [{(r['t'] - t0) / 60:.1f} min] {'hit' if r['hit'] else 'sight'}: {r['when']} -> {r['do']}"
                  f" reaction {r['reaction_s']}s (since the look {r['since_look_s']}s)")
        elif r.get("kind") == "goal":
            print(f"  goal done [{(r['t'] - t0) / 60:.1f} min] {r['goal']!r} {r['test']} in {r['secs']}s")
    says = [r["args"].get("text") for r in acts if r["tool"] == "say" and r["source"] == "model"]
    if says:
        print("  said in plans:", says[:8])


if __name__ == "__main__":
    main(*sys.argv[1:3])
