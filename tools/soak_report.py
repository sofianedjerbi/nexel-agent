"""Read a thinker's decisions.jsonl and report a soak: actions by source and kind, distance travelled, inventory
gained, stalls the watchdog declared, and every 20 s window with < 3 tiles of net progress and nothing gained.

    python tools/soak_report.py agents/gatsu0/decisions.jsonl
"""

import json
import re
import sys
from collections import Counter


def main(path):
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


if __name__ == "__main__":
    main(sys.argv[1])
