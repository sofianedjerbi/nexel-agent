"""Renders of the builds a saved live run holds, with the game's own renderer (#32, judge v3 part 2): each closed room
tools/judge_builds.py finds, seen as a player sees it (the HD look's camera, three quarters) and from high over it
(the same eye looking down at TOP_PITCH degrees), as small PNGs for the build review page (tools/build_review.py)
and for a vision model's rubric (tools/build_vlm.py). Drafts: ~/nexel/local/builds/renders/, outside the repos.

It plays a COPY of the run's QA save (rl_runs/score/<run>/saves/qa) on a game server of its own (nexel_worlds.server
--dev, a free port), and looks with Chrome (playwright: the `render` extra). Nobody is drawn: the eye is put on the
room (HdLook.eye), the player who looks stands RIGHT_OFF tiles south of it, out of the picture.

    python tools/render_build.py rl_runs/score/<run>                  # every room of the run
    python tools/render_build.py rl_runs/score/<run> --limit 3 --who Rook   # Rook's 3 best (v1+v2+v3)"""

import argparse
import base64
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nexel_worlds

import judge_builds as J

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "nexel" / "local" / "builds" / "renders"
W, H, ZOOM = 480, 320, 2              # (CSS px, one device px each; the land drawn ZOOM px an art px)
TOP_PITCH = 80                         # (degrees the eye looks down from over the room: the game's own camera goes to 58 at most)
RIGHT_OFF = 14                         # (tiles south of the room the looking player stands: past the picture's bottom)
VIEWS = {"three_quarter": False, "top": True}


def render_name(run: str, r: dict, view: str) -> str:
    """The file of a room's render: the run, the tile beside its door (judge_builds: "at"), the view."""
    return f"{run}_{r['at'][0]}_{r['at'][1]}_{view}.png"


def centre(m: dict) -> tuple:
    """The middle of a room's inside tiles (judge_builds.tiles: rows from x0, y0)."""
    pts = [(m["x0"] + i, m["y0"] + j) for j, row in enumerate(m["inside"]) for i, c in enumerate(row) if c == "1"]
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


# the eye on (x, y) at the land's top there; from over it (top): not free (a free eye keeps the surface's pitch), the
# look forced covered (hd.js api.force: the pitch eases to api.steep), steep TOP_PITCH
LOOK = """([x, y, top, steep]) => { const z = topAt(Math.round(x), Math.round(y));
  HdLook.eye = () => ({x, y, z, free: !top}); HdLook.force = top ? 1 : null; if (window.__steep === undefined) window.__steep = HdLook.steep;
  HdLook.steep = top ? steep : window.__steep; return 1; }"""
# the HD look's picture as it is (no overlay: the hotbar and words are not in it), a PNG data URL
PICTURE = """() => { const W = cv.width, H = cv.height, d = HdLook.read(0, 0, W, H).data, c = document.createElement('canvas'); c.width = W; c.height = H;
  const x = c.getContext('2d'), im = x.createImageData(W, H); im.data.set(d); for (let i = 3; i < d.length; i += 4) im.data[i] = 255; x.putImageData(im, 0, 0);
  return c.toDataURL('image/png'); }"""
SETTLED = "!HdLook.busy && HdLook.bare() === 0"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(qa: Path, work: Path):
    """A game server on a copy of the QA save -> (its process, port)."""
    db = qa / "world.sqlite"
    (work / "saves" / "qa").mkdir(parents=True)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as a, sqlite3.connect(work / "saves" / "qa" / "world.sqlite") as b:
        a.backup(b)
    seed = dict(sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("select key, value from meta")).get("seed", 12)
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "nexel_worlds.server", "--saves", str(work / "saves"), "--world", "qa",
                             "--port", str(port), "--seed", str(seed), "--dev"], cwd=Path(nexel_worlds.__file__).parents[2],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=lambda: os.nice(10))
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=.2).close()
            return proc, port
        except OSError:
            time.sleep(.1)
    proc.terminate()
    raise RuntimeError("the game server did not start")


def render(run: Path, rooms: list, out: Path) -> list:
    """Each room's views as PNGs in `out` -> the files written."""
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True)
    wrote = []
    with tempfile.TemporaryDirectory() as tmp:
        proc, port = serve(run / "saves" / "qa", Path(tmp))
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(channel="chrome")   # (the installed Google Chrome, as the game's tests: no browser to download)
                page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
                page.goto(f"http://127.0.0.1:{port}/?zoom={ZOOM}")
                page.wait_for_function("typeof ready !== 'undefined'", timeout=60000)
                page.click("#name")
                page.keyboard.type("renders")
                page.keyboard.press("Enter")
                page.wait_for_function("meObs && myPos()", timeout=60000)
                page.evaluate("() => { setOpt('names', false); setOpt('hints', false); setOpt('clouds', 0); rpc('dev_time', {phase: .35}); return 1; }")
                for r in rooms:
                    x, y = centre(r["map"])
                    page.evaluate("(a) => rpc('dev_teleport', a)", {"x": round(x), "y": round(y) + RIGHT_OFF})
                    page.wait_for_function(f"(() => {{ const p = myPos(); return p && p.y === {round(y) + RIGHT_OFF}; }})()", timeout=30000)
                    for view, top in VIEWS.items():
                        page.evaluate(LOOK, [x + .5, y + .5, top, TOP_PITCH])
                        page.wait_for_function(SETTLED, timeout=60000)
                        page.wait_for_function(f"Math.abs(HdLook.cam.pitch - {TOP_PITCH}) < .5" if top else f"HdLook.cam.pitch < {TOP_PITCH} - 15", timeout=30000)
                        page.evaluate("() => new Promise((ok) => { let n = 0; const f = () => ++n >= 30 ? ok(1) : requestAnimationFrame(f); f(); })")
                        page.wait_for_function(SETTLED, timeout=60000)
                        path = out / render_name(run.name, r, view)
                        path.write_bytes(base64.b64decode(page.evaluate(PICTURE).split(",", 1)[1]))
                        wrote.append(path)
                    print(f"{run.name}: room at {r['at']} ({r['by']}): {', '.join(VIEWS)}", flush=True)
                browser.close()
        finally:
            proc.terminate()
            proc.wait(10)
    return wrote


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="a saved live run (rl_runs/score/<run>: its saves/qa and the player's folder)")
    ap.add_argument("--who", help="only this one's rooms (the player's name, or 'players' for the real players')")
    ap.add_argument("--limit", type=int, help="only the best N rooms (by v1 + v2 + v3)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    from build_review import who_of           # (here: build_review imports this file for its renders)
    qa = a.run / "saves" / "qa"
    rooms = J.judged(qa / "world.sqlite", qa / "events.jsonl", who_of(a.run))
    rooms = sorted((r for r in rooms if not a.who or r["by"] == a.who), key=lambda r: -(r["v1"] + r["v2"] + r["v3"]))[:a.limit]
    for p in render(a.run, rooms, a.out):
        print(f"file://{p}")


if __name__ == "__main__":
    main()
