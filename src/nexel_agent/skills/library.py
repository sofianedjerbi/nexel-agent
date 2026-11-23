"""The skill library: one file per skill (agents/_skills/<name>.py, model-written) and index.json with what is known
of each (its docstring, how it is called, its own done_when test, who wrote it, its pass rate in the forge's
simulated worlds, its wins and losses in play). Shared by every player of the world."""

import json
import re
from pathlib import Path

DEMOTE_AFTER, DEMOTE_BELOW = 10, 0.5               # in play: at least this many runs, under this win rate: not offered


def holds(test: dict, obs) -> bool:
    """A skill's done_when on what the player sees now (placed: one within 3 tiles)."""
    from ..goals import missing
    from ..verbs import locate
    me = obs["self"]
    if "placed" in test:
        return any(max(abs(p[0] - me["x"]), abs(p[1] - me["y"])) <= 3 for p in locate(obs, str(test["placed"])))
    return not missing(test, obs)


class Library:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        f = self.path / "index.json"
        self.index: dict = json.loads(f.read_text()) if f.exists() else {}

    def save(self):
        (self.path / "index.json").write_text(json.dumps(self.index, indent=1))

    def sources(self) -> dict:
        return {n: (self.path / f"{n}.py").read_text() for n in self.index if (self.path / f"{n}.py").exists()}

    def add(self, name, src, call, done_when, author, sim_pass, gen_s=0.0):
        (self.path / f"{name}.py").write_text(src)
        doc = (re.search(r'"""(.*?)"""', src, re.S) or [None, ""])[1].strip().splitlines()
        self.index[name] = {"doc": (doc[0] if doc else "")[:120], "call": call, "done_when": done_when, "author": author,
                            "sim": sim_pass, "wins": 0, "losses": 0, "gen_s": round(gen_s, 1)}
        self.save()

    def offered(self) -> list[str]:
        """The skills a plan may use: all but the demoted ones."""
        out = []
        for n, m in self.index.items():
            runs = m.get("wins", 0) + m.get("losses", 0)
            if runs >= DEMOTE_AFTER and m.get("wins", 0) / runs < DEMOTE_BELOW:
                continue
            out.append(n)
        return out

    def lines(self, k=12) -> str:
        """For the plan prompt: 'skill get_wood N (play 3/4), skill make_workbench, ...' (names say what they do)."""
        def one(n):
            m = self.index[n]
            args = " ".join(a.upper() for a in m.get("call", "").split(",") if a.strip())
            runs = m["wins"] + m["losses"]
            test = m.get("done_when") or {}
            done = (f"{next(iter(test['have']))} N" if args and len(test.get("have", {})) == 1 else
                    json.dumps(test, separators=(",", ":")))
            return f"{n}{' ' + args if args else ''} [done: {done}]" + (f" (won {m['wins']}/{runs})" if runs else "")
        return ", ".join(one(n) for n in self.offered()[:k])

    def outcome(self, name, won: bool):
        if name in self.index:
            self.index[name]["wins" if won else "losses"] += 1
            self.save()
