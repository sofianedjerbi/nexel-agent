"""Skills the model writes run here (design 3.1): one Python function each, against a small API and nothing else.

    act(command) -> reply      one command in the plan language: "dig_at tree", "craft workbench", "place_near chest",
                               "craft_here axe", "fight cow", "move_to 3 -4"... ({"ok", "error", "detail", "result"})
    look() -> obs              what the player sees (beings carry absolute "x", "y")
    me() -> dict               look()["self"];   have(item) -> int   how many you carry
    find(name) -> [(x, y)]     where things of that name are in view, nearest first
    and the other skills of the library, by name.

No imports, no dunders, no open/eval/exec, safe builtins only, a budget of actions and seconds. This runs code; it
decides nothing: what to do is in the skill, which the model wrote.
"""

import builtins
import math
import sys
import time
import traceback

SAFE = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map",
    "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "print", "next", "iter",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "StopIteration", "True", "False", "None")}
BANNED = ("import ", "__", "open(", "eval(", "exec(", "compile(", "globals(", "locals(")


class Budget(Exception):
    pass


class Stopped(Exception):
    pass


def check(name, src) -> str | None:
    """None when src compiles and defines `name`, else what is wrong."""
    bad = next((b for b in BANNED if b in src), None)
    if bad:
        i = src.find(bad)
        line = src[src.rfind("\n", 0, i) + 1:src.find("\n", i) if src.find("\n", i) != -1 else None].strip()
        return f"not allowed: {line[:60]!r} (no imports, dunders, open, eval, exec: use the API and str methods)"
    try:
        g = {"__builtins__": SAFE}
        exec(compile(src, f"<{name}>", "exec"), g)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} at line {e.lineno}: {(e.text or '').strip()}"
    except Exception as e:                                # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None if callable(g.get(name)) else f"the code must define a function named {name}"


class Sandbox:
    """do(command) -> reply and observe() -> obs are the body's (in-process sim for the forge, the live game in play)."""

    def __init__(self, do, observe, max_actions=300, max_s=120.0, stop=lambda: False):
        self.do, self.observe = do, observe
        self.max_actions, self.max_s, self.stop = max_actions, max_s, stop
        self.trace: list[str] = []

    def _act(self, command, *more):
        if self.stop():
            raise Stopped("stopped (a card fired)")
        if len(self.trace) >= self.max_actions or time.monotonic() - self.t0 > self.max_s:
            raise Budget(f"budget spent ({len(self.trace)} actions, {time.monotonic() - self.t0:.0f} s)")
        cmd = " ".join(str(x) for x in (command, *more))
        r = self.do(cmd)
        short = r.get("error") and f"FAILED {r.get('error')}: {str(r.get('detail'))[:90]}" or "ok " + str(
            ((r.get("result") or {}).get("changed") or {}).get("inventory") or (r.get("result") or {}).get("text") or "")[:90]
        self.trace.append(f"{cmd} -> {short}")
        return r

    def _look(self):
        o = self.observe()
        for e in o["entities"]:
            e["x"], e["y"] = o["self"]["x"] + e["dx"], o["self"]["y"] + e["dy"]
        return o

    def _find(self, name):
        from ..verbs import locate
        return locate(self.observe(), str(name))

    def env(self, library: dict):
        g = {"__builtins__": SAFE, "act": self._act, "look": self._look, "me": lambda: self.observe()["self"],
             "have": lambda item: self.observe()["self"]["inventory"].get(str(item), 0), "find": self._find, "math": math}
        for src in library.values():
            exec(compile(src, "<library>", "exec"), g)
        return g

    def run(self, name, src, args=(), library=None) -> dict:
        """Run skill `name` (src, plus the library). {value, error, trace, seconds}."""
        self.trace, self.t0 = [], time.monotonic()
        err = check(name, src)
        if err:
            return {"value": None, "error": err, "trace": [], "seconds": 0.0}
        g = self.env({k: v for k, v in (library or {}).items() if k != name})
        exec(compile(src, f"<{name}>", "exec"), g)
        deadline = self.t0 + self.max_s

        def tracer(frame, event, arg):                 # (a loop that never acts still meets the time budget)
            if not frame.f_code.co_filename.startswith("<"):
                return None                           # (only the skill's own lines are watched)
            if time.monotonic() > deadline:
                raise Budget(f"budget spent ({self.max_s:.0f} s, {len(self.trace)} actions)")
            return tracer
        old = sys.gettrace()
        sys.settrace(tracer)
        try:
            value = g[name](*args)
            error = None
        except (Budget, Stopped) as e:
            value, error = None, str(e)
        except Exception as e:                            # noqa: BLE001  (the model's code: any error is an outcome)
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f"line {f.lineno}: {f.line}" for f in reversed(tb) if f.filename.startswith("<")), "")
            value, error = None, f"{type(e).__name__}: {e} ({where})"
        finally:
            sys.settrace(old)
        return {"value": value, "error": error, "trace": self.trace, "seconds": round(time.monotonic() - self.t0, 2)}
