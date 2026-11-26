"""The brain feed: what a live player sees, thinks, decides and remembers, one JSON object a line (docs/brain_feed.md),
appended to a file and served as Server-Sent Events on 127.0.0.1:PORT/feed for the brain view (tools/brain/index.html).

Stdlib only, and the body never waits on it: see() reads a few fields and puts them on a queue (a frame at most once a
second); one thread turns them into lines (with the policy's six likeliest actions: one more forward pass, there),
writes the file and wakes the pages. The mind (rl/mind.py) is watched by wrapping its facts() and apply(): what it was
told and what it answered, with no line added there.
    python -m nexel_agent.rl.live ... --brain 8790        then open tools/brain/index.html?feed=http://127.0.0.1:8790/feed"""

import json
import math
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .facts import HOSTILE, Attack, Craft, Say, read, told

V = 1
SIDE = 13                      # the tiles round it: the middle 13 x 13 of the agent API's grid
FRAME_S = 1.0                  # a frame at most this often
TOP = 6
BEINGS = 8
BACKLOG = 400                  # lines kept for a page that connects late
PING_S = 15.0


def crop(grid: list, side: int = SIDE) -> list:
    """The middle side x side of the grid (rows of characters, itself in the middle)."""
    if not grid:
        return []
    o = (len(grid) - side) // 2
    return [row[o:o + side] for row in grid[o:o + side]] if o >= 0 else list(grid)


def frame(obs: dict, facing: str = "S", act: str | None = None, mask=None, goal=None, since_s: float | None = None,
          allies=(), foes=()) -> dict:
    """A frame from one observation (the agent API's) and what the body knows: goal (rl/goals.Goal or None), allies
    (names the mind calls allies), foes (ids that hit it)."""
    me = obs.get("self", {})
    f = {"k": "frame", "pos": [me.get("x"), me.get("y")] + ([me["z"]] if me.get("z") is not None else []),
         "hp": me.get("hp"), "food": me.get("hunger"), "facing": facing}
    if obs.get("time"):
        f["time"] = obs["time"]
    if obs.get("grid"):
        f["tiles"] = crop(obs["grid"])
    beings = []
    for e in obs.get("entities", [])[:BEINGS]:
        b = {k: e[k] for k in ("name", "kind", "dx", "dy", "hp") if e.get(k) is not None}
        side = "ally" if e.get("name") in allies else "foe" if e.get("kind") in HOSTILE or e.get("id") in foes else None
        if side:
            b["side"] = side
        beings.append(b)
    f["beings"] = beings
    f["inv"] = dict(me.get("inventory") or {})
    if goal is not None and getattr(goal, "kind", "free") != "free":
        g = {"kind": goal.kind}
        if goal.item:
            g["item"] = goal.item
            g["have"] = f["inv"].get(goal.item, 0)
        if goal.person:
            g["with"] = goal.person
        if since_s is not None:
            g["since_s"] = round(since_s)
        f["goal"] = g
    if act:
        f["act"] = act
    if mask is not None:
        f["legal"], f["of"] = int(sum(bool(m) for m in mask)), len(mask)
    return f


def top(logits, mask=None, names=None, n: int = TOP) -> list:
    """The n likeliest actions after the mask, [[name, p], ...] (softmax of the logits; plain lists or numpy)."""
    lg = [float(x) for x in logits]
    if mask is not None:
        lg = [x if (i < len(mask) and mask[i]) else -math.inf for i, x in enumerate(lg)]
    hi = max(lg)
    ex = [math.exp(x - hi) if x > -math.inf else 0.0 for x in lg]
    s = sum(ex) or 1.0
    best = sorted(range(len(ex)), key=lambda i: -ex[i])[:n]
    return [[names(i) if names else str(i), round(ex[i] / s, 3)] for i in best if ex[i] > 0]


def chat_or_event(e: dict, me: str) -> dict | None:
    """A line of the feed from one game event (or None: not one the page shows). The words: rl/facts.py told."""
    ev = read(e)
    if isinstance(ev, Say):
        return {"k": "chat", "from": ev.src, "text": ev.text[:200], **({"said": True} if ev.src == me else {})}
    if ev is None or (isinstance(ev, Craft) and ev.name != me):
        return None
    if isinstance(ev, Attack):
        return {"k": "event", "what": "hurt" if ev.target == me else "attack", "text": f"{told(ev, str)} ({ev.damage:g})",
                **({"n": ev.damage} if ev.damage else {})}
    return {"k": "event", "what": e["type"], "text": told(ev, str)}


class Brain:
    def __init__(self, port: int | None = None, path=None, name: str = "", persona: str = "", host: str = "127.0.0.1"):
        """port: serve /feed there (0: any free port, see .port; None: no server). path: append the lines there."""
        self.name, self.q, self.cond = name, queue.SimpleQueue(), threading.Condition()
        self.lines: deque = deque(maxlen=BACKLOG)      # (seq, line)
        self.seq, self.sticky = 0, {}                    # hello, memory, tally: always sent first to a page that connects
        self.file = open(path, "a") if path else None
        self.t_frame, self.mind, self.asked, self.deaths, self.tally = 0.0, None, None, None, None
        self.server, self.port = None, None
        threading.Thread(target=self._run, daemon=True, name="brain").start()
        if port is not None:
            self.server = ThreadingHTTPServer((host, port), _handler(self))
            self.server.daemon_threads = True
            self.port = self.server.server_address[1]
            threading.Thread(target=self.server.serve_forever, daemon=True, name="brain-http").start()
        self.emit({"k": "hello", "v": V, "name": name, "persona": persona})

    # ------------------------------------------------------------------ from the body (never waits)
    def emit(self, line: dict, job=None):
        line.setdefault("t", round(time.time(), 1))
        self.q.put((line, job))

    def see(self, obs: dict, body, enc=None, mask=None, i: int | None = None):
        """After each decision of the live body (rl/live.py): its events now, a frame if a second went by."""
        try:
            self.attach(getattr(body, "mind", None))
            me = obs.get("self", {})
            hit = False
            for e in obs.get("events", []):
                line = chat_or_event(e, self.name)
                if line:
                    self.emit(line)
                    hit |= line.get("what") == "hurt"
            if self.deaths is not None and me.get("deaths", 0) > self.deaths:
                self.emit({"k": "event", "what": "death", "text": f"{self.name} died"})
            self.deaths = me.get("deaths", 0)
            if getattr(body, "hurt", 0) > 0 and not hit:        # (hurt by no one: a fall, hunger)
                self.emit({"k": "event", "what": "hurt", "text": f"{self.name} lost {body.hurt:g} health", "n": body.hurt})
            now = time.monotonic()
            if now - self.t_frame < FRAME_S:
                return
            self.t_frame = now
            from .encode import action_name
            mind = self.mind
            self.tried(mind)
            rel = mind.relations if mind else {}
            f = frame(obs, getattr(body, "facing", "S"), action_name(i) if i is not None else None, mask,
                      mind.goal if mind else None, now - mind.goal_at if mind else None,
                      {n for n, r in rel.items() if r.get("ally")}, set(getattr(body, "foes", {})))
            pol = getattr(body, "pol", None)
            self.emit(f, (pol, enc, mask, action_name) if pol is not None and enc is not None else None)
        except Exception:                                # (a view must never stop the player)
            pass

    def tried(self, mind):
        """How each goal the mind set went this session (mind.tried: {goal: [set, done]}), as a `tally` line when it
        changed: a goal's end is judged in the mind's own step, not in a call the brain wraps, so it is looked at once a
        frame."""
        t = getattr(mind, "tried", None)
        if not t:
            return
        s = json.dumps(t, sort_keys=True)
        if s != self.tally:
            self.tally = s
            self.emit({"k": "tally", "goals": json.loads(s)})

    def attach(self, mind):
        """Watch the mind: what it is told (facts) and what it answers (apply), as `thought` and `memory` lines."""
        if mind is None or mind is self.mind:
            return
        self.mind, facts, apply = mind, mind.facts, mind.apply

        def facts_(obs):
            s = facts(obs)
            if self.asked is None or time.monotonic() - self.asked[1] > 60:     # (the first call of a thought)
                self.asked = (s, time.monotonic())
            return s

        def apply_(d):
            before, goal = json.dumps(mind.relations, sort_keys=True), mind.goal
            apply(d)
            g = mind.goal
            if g is not goal:
                self.emit({"k": "event", "what": "goal", "text": "no goal" if g.kind == "free" else "goal: " + " ".join(
                    str(p) for p in (g.kind, g.item, g.person and f"(with {g.person})") if p)})
            s, t0 = self.asked or ("", None)
            self.asked = None
            line = {"k": "thought", "facts": s, "reply": d}
            if t0 is not None:
                line["took_s"] = round(time.monotonic() - t0, 2)
            self.emit(line)
            if json.dumps(mind.relations, sort_keys=True) != before:
                self.emit({"k": "memory", "people": json.loads(json.dumps(mind.relations))})

        mind.facts, mind.apply = facts_, apply_
        self.emit({"k": "memory", "people": json.loads(json.dumps(mind.relations))})

    # ------------------------------------------------------------------ the thread
    def _run(self):
        while True:
            line, job = self.q.get()
            if line is None:
                break
            if job is not None:
                pol, enc, mask, names = job
                try:
                    line["top"] = top(pol.logits(enc), mask, names)
                    if hasattr(pol, "value"):
                        line["value"] = round(float(pol.value(enc)), 3)
                except Exception:
                    pass
            text = json.dumps(line, separators=(",", ":"))
            if self.file:
                self.file.write(text + "\n")
                self.file.flush()
            with self.cond:
                self.seq += 1
                if line["k"] in ("hello", "memory", "tally"):
                    self.sticky[line["k"]] = (self.seq, text)
                self.lines.append((self.seq, text))
                self.cond.notify_all()

    def flush(self, timeout: float = 2.0):
        """Wait till every line put so far is out (tests, and before closing)."""
        end = time.monotonic() + timeout
        while not self.q.empty() and time.monotonic() < end:
            time.sleep(0.005)
        time.sleep(0.01)

    def close(self):
        self.flush()
        self.q.put((None, None))
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.file:
            self.file.close()

    def since(self, seq: int) -> list:
        """The lines after seq (a page's first call: seq -1, the sticky ones first)."""
        with self.cond:
            out = [ln for ln in self.lines if ln[0] > seq]
            if seq < 0:
                first = sorted(v for v in self.sticky.values() if not out or v[0] < out[0][0])
                out = first + out
            return out


def sse(text: str) -> bytes:
    """One Server-Sent Event (a line of the feed is one JSON object: no newline in it)."""
    return f"data: {text}\n\n".encode()


def _handler(brain: Brain):
    class Feed(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self):
            if self.path.split("?")[0] != "/feed":
                self.send_error(404)
                return
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seq = -1
            try:
                while True:
                    out = brain.since(seq)
                    if out:
                        self.wfile.write(b"".join(sse(t) for _, t in out))
                        seq = max(seq, out[-1][0])
                    else:
                        with brain.cond:                 # (wait under the lock, write outside it)
                            woke = brain.seq > seq or brain.cond.wait(PING_S)
                        if not woke:
                            self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
    return Feed
