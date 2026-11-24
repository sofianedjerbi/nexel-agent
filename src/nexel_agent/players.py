"""One command for several AI players: the shared local model server plus N trained players (rl/live.py: the
shipped body models/rook.npz acts, the model plans and talks), each with a persona.

    nexel-agent-players                      # 2 players, the local game at :7777
    nexel-agent-players --n 4 --model 30b
    nexel-agent-players --pick gatsu0,senku0 --minutes 30
    nexel-agent-players --list               # the roster

Starts mlx_lm.server on --port (or reuses one already answering there), waits until it answers, then
starts the players a few seconds apart, each logging to <data>/<name>.log and <data>/<name>.rl.jsonl. Prints
one line per player as it acts. Ctrl-C / SIGTERM / a crash stop every child and the model server it started:
nothing is left holding RAM.
"""

import argparse
import atexit
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROSTER = Path(__file__).with_name("personas.json")
MODELS = {
    "4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "8b": "mlx-community/Qwen3-8B-4bit",
    "14b": "mlx-community/Qwen3-14B-4bit",
    "30b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
}
DEFAULT_MODEL = "auto"    # 30b when the Mac has room for it, else 4b (bench 2026-11-23: the 30B talks and plans best; the 8B breaks the planner's JSON)
AUTO_FREE_PCT = 70        # free memory before loading the 30B (~17 GB of 32): what stays free is then about 20%
SRC = str(Path(__file__).resolve().parent.parent)    # so children find the package without an install


def load_roster(path: Path = ROSTER) -> list[dict]:
    return json.loads(Path(path).read_text())


def choose(roster: list[dict], n: int, pick: str | None, seed: int | None = None) -> list[dict]:
    """The named players (--pick, in that order), else n distinct ones at random."""
    if pick:
        by = {p["name"]: p for p in roster}
        names = [s.strip() for s in pick.split(",") if s.strip()]
        missing = [s for s in names if s not in by]
        if missing:
            raise SystemExit(f"not in the roster: {', '.join(missing)} (see --list)")
        return [by[s] for s in dict.fromkeys(names)]
    if not 1 <= n <= len(roster):
        raise SystemExit(f"--n must be 1..{len(roster)}")
    return random.Random(seed).sample(roster, n)


def free_pct(run=subprocess.run) -> int | None:
    """The system's free memory in % (macOS memory_pressure), or None when it cannot be read."""
    try:
        out = run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
        return int(re.search(r"free percentage:\s*(\d+)%", out).group(1))
    except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
        return None


def model_name(m: str, free=None) -> str:
    if m.lower() == "auto":
        f = free_pct() if free is None else free
        return MODELS["30b" if f is not None and f >= AUTO_FREE_PCT else "4b"]
    return MODELS.get(m.lower(), m)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nexel-agent-players", description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=2, help="how many players (ignored with --pick); each one is a Python process and the one model server serves them in turn: more than ~4 loads this Mac hard")
    p.add_argument("--pick", help="comma-separated roster names, e.g. gatsu0,senku0")
    p.add_argument("--game", default="ws://127.0.0.1:7777/agent", help="the game's agent websocket")
    p.add_argument("--minutes", type=float, default=None, help="stop after this long (default: run until Ctrl-C)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"one of {', '.join(MODELS)} or a full HF / local model path (default {DEFAULT_MODEL})")
    p.add_argument("--port", type=int, default=8080, help="the shared model server's port")
    p.add_argument("--stagger", type=float, default=4.0, help="seconds between player starts")
    p.add_argument("--data", default="agents", help="where notepads and logs go")
    p.add_argument("--seed", type=int, default=None, help="random pick seed")
    p.add_argument("--list", action="store_true", help="print the roster and exit")
    p.add_argument("--policy", default="", help="the body's weights (.npz from rl/export.py; default: the shipped models/rook.npz)")
    return p.parse_args(argv)


def game_http(ws_url: str) -> str:
    u = urlsplit(ws_url)
    scheme = "https" if u.scheme in ("wss", "https") else "http"
    return f"{scheme}://{u.netloc}/"


def reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        httpx.get(url, timeout=timeout)
        return True
    except httpx.HTTPError:
        return False


def model_up(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def last_event(log_text: str) -> str | None:
    """The newest thing worth showing from a player's log (rl/live.py: a JSON line per decision): what it did, where."""
    for line in reversed(log_text.splitlines()[-50:]):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and "act" in d:
            return f"{d['act']} at {tuple(d.get('pos') or ())}, health {d.get('hp')}"
    return None


class Launcher:
    def __init__(self, args, popen=subprocess.Popen, out=print):
        self.a, self.popen, self.out = args, popen, out
        self.data = Path(args.data)
        self.model_proc = None
        self.players: dict[str, object] = {}
        self.shown: dict[str, str] = {}
        self._stopped = False

    # ------------------------------------------------------------ start
    def start_model(self, wait_s: float = 900) -> None:
        if model_up(self.a.port):
            self.out(f"model server: reusing the one on :{self.a.port}")
            return
        name = model_name(self.a.model)
        self.out(f"model server: starting {name} on :{self.a.port} (log {self.data / 'model_server.log'})")
        log = open(self.data / "model_server.log", "a")
        self.model_proc = self.popen(
            [sys.executable, "-m", "mlx_lm.server", "--model", name, "--host", "127.0.0.1",
             "--port", str(self.a.port), "--chat-template-args", '{"enable_thinking":false}'],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.monotonic()
        while not model_up(self.a.port):
            if self.model_proc.poll() is not None:
                raise SystemExit(f"model server exited ({self.model_proc.returncode}); see {self.data / 'model_server.log'}")
            if time.monotonic() - t0 > wait_s:
                raise SystemExit("model server did not answer in time")
            time.sleep(1)
        self.warm()
        self.out(f"model server: up after {time.monotonic() - t0:.0f}s")

    def warm(self) -> None:
        """mlx_lm.server loads the weights on the first request: do it once here, not in four players at once."""
        try:
            httpx.post(f"http://127.0.0.1:{self.a.port}/v1/chat/completions", timeout=600, json={
                "model": "default_model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})
        except httpx.HTTPError:
            pass

    def player_cmd(self, p: dict) -> list[str]:
        cmd = [sys.executable, "-m", "nexel_agent.rl.live", "--name", p["name"], "--persona", p["persona"],
               "--game", self.a.game, "--model-url", f"http://127.0.0.1:{self.a.port}/v1/chat/completions",
               "--log", str(self.data / f"{p['name']}.rl.jsonl"), *(["--policy", self.a.policy] if self.a.policy else [])]
        return cmd + (["--minutes", str(self.a.minutes)] if self.a.minutes else [])

    def start_player(self, p: dict) -> None:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])))
        log = open(self.data / f"{p['name']}.log", "a")
        self.players[p["name"]] = self.popen(self.player_cmd(p), stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        self.out(f"{p['name']:<12} starting ({p['role']}: {p['ref']})")

    # ------------------------------------------------------------- watch
    def report(self) -> None:
        for name, proc in self.players.items():
            code = proc.poll()
            if code is not None:
                f = self.data / f"{name}.log"
                tail = f.read_text(errors="replace").strip().splitlines()[-1:] if f.exists() else []
                msg = f"exited ({code})" + (f": {tail[0][-100:]}" if code and tail else "")
            else:
                f = self.data / f"{name}.rl.jsonl"
                msg = last_event(f.read_text(errors="replace")) if f.exists() else None
            if msg and self.shown.get(name) != msg:
                self.shown[name] = msg
                self.out(f"{name:<12} {msg}")

    def alive(self) -> bool:
        return any(p.poll() is None for p in self.players.values())

    # -------------------------------------------------------------- stop
    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        procs = [p for p in self.players.values() if p.poll() is None]
        if self.model_proc is not None and self.model_proc.poll() is None:
            procs.append(self.model_proc)
        for p in procs:
            p.terminate()
        deadline = time.monotonic() + 8
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        if procs:
            self.out(f"stopped {len(procs)} process(es)" + (" incl. the model server" if self.model_proc else ""))

    def run(self, chosen: list[dict]) -> int:
        url = game_http(self.a.game)
        if not reachable(url):
            self.out(f"The game server is not answering at {url}\n"
                     f"Start it first (python -m nexel_worlds.server) or pass --game ws://HOST:PORT/agent.")
            return 2
        self.data.mkdir(parents=True, exist_ok=True)
        atexit.register(self.stop)
        try:
            self.start_model()
            for i, p in enumerate(chosen):
                if i:
                    self._sleep_watching(self.a.stagger)
                self.start_player(p)
            while self.alive():
                self._sleep_watching(3)
            self.report()
            self.out("all players finished")
            return 0
        except KeyboardInterrupt:
            self.out("\nstopping...")
            return 130
        finally:
            self.stop()

    def _sleep_watching(self, s: float) -> None:
        end = time.monotonic() + s
        while time.monotonic() < end:
            time.sleep(min(1.0, max(0.0, end - time.monotonic())))
            self.report()


def _sigterm(signum, frame):
    raise KeyboardInterrupt


def main(argv=None) -> int:
    a = parse_args(argv)
    roster = load_roster()
    if a.list:
        for p in roster:
            print(f"{p['name']:<12} {p['role']:<15} {p['persona']}")
        return 0
    chosen = choose(roster, a.n, a.pick, a.seed)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGHUP, _sigterm)
    return Launcher(a).run(chosen)


if __name__ == "__main__":
    sys.exit(main())
