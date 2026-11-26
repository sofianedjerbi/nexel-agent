"""One command for the live scorecard (#2): a QA copy of a world (never the one on :7777), a --dev server on it, the
AI player for N minutes with the scripted visitors (tools/rl_social_soak.py: Mira, who talks, asks and reacts to a
blow; Brute, who sets upon her), then the table (tools/scorecard.py) beside the real people of the source world.

    python tools/rl_live_scorecard.py --minutes 30                       # words and plans from the model on :8080
    python tools/rl_live_scorecard.py --minutes 10 --model-url ""        # the body alone, in silence

Everything goes to rl_runs/score/<time>/: the world copy and its events.jsonl, the player's logs (rl.jsonl,
mind.jsonl, latency.jsonl, brain.jsonl), the visitor's account (visitor.json), scorecard.md.

The model on :8080: `python -m mlx_lm.server --model <it> --port 8080`, or players.py. Its defaults (mlx-lm 0.32.2)
decode the requests in hand together (--decode-concurrency 32), which a reply going beside a thought counts on
(rl/voice.py, #11): it does so for a model whose cache can merge, with no draft model, no --kv-bits and no seed in
the request. Without that, a reply waits for the thought at the server: latency.jsonl's llm_s says so."""

import argparse
import json
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME_SAVES = ROOT.parent / "nexel-worlds" / "saves"


def copy_world(src: Path, saves: Path) -> Path:
    """The world's land, copied whole while its server may be writing it (sqlite's own backup); a fresh event log."""
    dst = saves / "qa"
    dst.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{src / 'world.sqlite'}?mode=ro", uri=True) as a, sqlite3.connect(dst / "world.sqlite") as b:
        a.backup(b)
    return dst


def up(port: int, timeout=60.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", default=str(GAME_SAVES / "caves"), help="the world to copy (its events.jsonl: the real people)")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--port", type=int, default=7792)
    ap.add_argument("--name", default="Rook")
    ap.add_argument("--policy", default="", help="another body's weights (default: the shipped models/rook.npz)")
    ap.add_argument("--model-url", default=None, help="the words and the planner (default: the player's own); \"\" for none")
    ap.add_argument("--no-visitors", action="store_true")
    a = ap.parse_args()
    if a.port == 7777:
        sys.exit("never on :7777: that is the live world (#16)")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", a.port)) == 0:
            sys.exit(f"something already answers on :{a.port}")
    src, out = Path(a.world), ROOT / "rl_runs" / "score" / time.strftime("%Y%m%d-%H%M%S")
    world = copy_world(src, out / "saves")
    logs = out / a.name
    logs.mkdir(parents=True, exist_ok=True)
    py, procs = sys.executable, []
    try:
        procs.append(subprocess.Popen([py, "-m", "nexel_worlds.server", "--port", str(a.port), "--world", "qa",
                                       "--saves", str(out / "saves"), "--dev"], cwd=out,
                                      stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT))
        if not up(a.port):
            sys.exit(f"the QA server did not come up: {out / 'server.log'}")
        cmd = [py, "-m", "nexel_agent.rl.live", "--name", a.name, "--game", f"ws://127.0.0.1:{a.port}/agent",
               "--minutes", str(a.minutes), "--log", str(logs / "rl.jsonl"), "--memory", str(logs / "memory.json"),
               "--brain", str(a.port + 1)]          # (brain.jsonl beside rl.jsonl: the replay for tools/brain/index.html)
        cmd += ["--policy", a.policy] if a.policy else []
        cmd += ["--model-url", a.model_url] if a.model_url is not None else []
        player = subprocess.Popen(cmd, cwd=ROOT, stdout=open(logs / "report.json", "w"), stderr=open(logs / "stderr.log", "w"))
        if not a.no_visitors:
            time.sleep(5)                                   # (it is in the world before they come)
            procs.append(subprocess.Popen([py, str(ROOT / "tools" / "rl_social_soak.py"), "--port", str(a.port), "--target",
                                           a.name, "--minutes", str(a.minutes)], cwd=ROOT,
                                          stdout=open(out / "visitor.json", "w"), stderr=open(out / "visitor.log", "w")))
        player.wait()
        for p in procs[1:]:                                 # (the visitors finish their minutes and tell what they saw)
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    card = subprocess.run([py, str(ROOT / "tools" / "scorecard.py"), "--events", str(world / "events.jsonl"), "--who", a.name,
                           "--logs", str(logs), "--humans", str(src / "events.jsonl")], capture_output=True, text=True)
    text = card.stdout + card.stderr + visited(out / "visitor.json")
    (out / "scorecard.md").write_text(text)
    subprocess.run([py, str(ROOT / "tools" / "kpis.py"), "live", str(out)], capture_output=True)    # (its row in docs/kpis.md)
    print(text + f"\n(all of it in {out}; its KPIs: docs/kpis.md)")


def visited(path: Path) -> str:
    """What the visitors saw it do (tools/rl_social_soak.py): asks of Mira's it met, its own asks, Brute's fights."""
    try:
        v = json.loads(path.read_text())
    except (OSError, ValueError):
        return ""
    asks = v.get("asks", [])
    met = [x for x in asks if x.get("got") is not None]
    could = [x for x in asks if x.get("it_had")]
    rows = [("Mira's asks it met", f"{len(met)} of {len(asks)} ({len(could)} it could)"),
            ("its own asks of Mira", len(v.get("its_asks", []))),
            ("first strikes on Mira", len(v.get("first_strikes", []))),
            ("Brute fights it joined", f"{sum(b.get('brute_hp', [0, 0])[0] != b.get('brute_hp', [0, 0])[1] for b in v.get('brute', []))} "
                                       f"of {len(v.get('brute', []))}")]
    return "\n\n| the visitors saw | |\n|---|---|\n" + "\n".join(f"| {k} | {x} |" for k, x in rows) + "\n"


if __name__ == "__main__":
    main()
