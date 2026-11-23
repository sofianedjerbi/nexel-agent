"""The skill forge (design stage 2): offline, one model, at night. For each curriculum goal the model writes one skill
(it reasons in a scratch block first, then the code), the skill runs in fresh simulated worlds (simworld.py), its
own done_when test says pass or fail, a failure goes back with the error, the last actions and the state, and a
skill that passes in at least PASS of the worlds joins the shared library (library.py). The goals and their tests
are the game's curriculum (facts); every line of a skill is the model's.

    python -m megastructure.skills.forge --model-url http://127.0.0.1:8091/v1/chat/completions --lib agents/_skills
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx

from ..goals import missing
from ..tiles import RECIPES
from .library import Library
from .sandbox import Sandbox, check
from .simworld import SimBody

SEEDS = (1, 5, 7, 9, 10)                     # generated worlds with trees and rocks within 40 tiles of the spawn
PASS = 4
TASKS = [  # (name, call args, what it is for, done_when) -- the curriculum, in order
    ("get_wood", (6,), "get at least n wood into your pack", {"have": {"wood": 6}}),
    ("get_stone", (3,), "get at least n stone into your pack", {"have": {"stone": 3}}),
    ("get_fiber", (2,), "get at least n fiber into your pack", {"have": {"fiber": 2}}),
    ("make_workbench", (), "end with a workbench placed within 2 tiles of you", {"placed": "workbench"}),
    ("make_axe", (), "end with an axe in your pack", {"have": {"axe": 1}}),
    ("make_pickaxe", (), "end with a pickaxe in your pack", {"have": {"pickaxe": 1}}),
    ("make_hoe", (), "end with a hoe in your pack", {"have": {"hoe": 1}}),
    ("make_campfire", (), "end with a campfire placed within 2 tiles of you", {"placed": "campfire"}),
    ("make_wood_wall", (), "end with a wood_wall in your pack", {"have": {"wood_wall": 1}}),
    ("make_door", (), "end with a door in your pack", {"have": {"door": 1}}),
    ("make_chest", (), "end with a chest in your pack", {"have": {"chest": 1}}),
]
API = """You write skills for a player in a top-down survival game (grid, north is y-1, absolute x,y). A skill is ONE
Python function. It may use only these (no imports, no files):
  act(command) -> dict    one command, waits for it to end. Reply: {"ok": bool, "error": str, "detail": str,
                          "result": {"changed": {"inventory": {item: +n}, "pos": [x, y]}, ...}}
  look() -> dict          {"self": {"x","y","hp","hunger","inventory": {item: n}}, "entities": [{"id","kind","x","y","hp"}], ...}
  me() -> dict            look()["self"]
  have(item) -> int       how many of item you carry
  find(name) -> [(x, y)]  where things of that name are in view (21x21 around you), nearest first: find("tree")
  math, and the library skills listed below (call them by name).
Commands for act (verbs do the walking for you):
  "dig_at tree" / "dig_at rock" / "dig_at bare_bush" / "dig_at x y" [+ " with axe"]: walk next to the nearest one
      (in view, or where one was last seen) and dig it. One dig takes a whole tree (3 wood), rock (2 stone), bush.
  "craft NAME [N]": craft from your pack. "craft_here NAME": the same, walking to a placed station first if one is near.
  "place_near ITEM": put ITEM down on a free flat tile next to you (or the nearest reachable one).
  "go_next_to TARGET", "move_to X Y", "walk N|E|S|W" (keeps walking that way), "look_far N|E|S|W", "fight TARGET", "eat ITEM".
  A failed act has "ok": False and says why in "error"/"detail"; nothing in view: "error": "not_found".
  "look_far D" reports what lies up to 58 tiles that way: reply["result"]["things"] = [{"name": "tree", "x", "y",
  "dist"}, ...] nearest first (names: tree, rock, bare bush, berry bush, ore, water...), and
  the player remembers it: after it, "dig_at tree" walks to a tree look_far saw, even one far out of view.
  A start may have no tree or rock in view: the world is large, things are found by looking and walking.
Facts: {recipes}. A craft marked @workbench needs a placed workbench within 2 tiles of you; craft_here walks there
if one is in view. Wood and stone cannot be placed. Bare bushes and thickets give fiber. Ore needs a pickaxe.
Answer: first your reasoning in a ```text block (the steps, what can go wrong, how you check), then the code in one
```python block: def <name>(...): with a one-line docstring; check replies and look again after acting; return True
when done, False when it cannot be done. Never call a variable act, look, me, have or find."""


def recipes_text():
    names = ("workbench", "axe", "pickaxe", "hoe", "campfire", "wood_wall", "door", "chest", "plank", "fence")
    return "; ".join(f"{n} = " + " + ".join(f"{c} {k}" for k, c in RECIPES[n].inputs.items())
                     + (f" @{RECIPES[n].station}" if RECIPES[n].station else "") for n in names)


def code_of(text, name="") -> str:
    """The code block of the answer (the last fenced block that defines the function; never the reasoning)."""
    blocks = re.findall(r"```[a-zA-Z]*[ \t]*\n(.*?)(?:```|$)", text, re.S)
    with_def = [b for b in blocks if f"def {name}" in b] or [b for b in blocks if "def " in b]
    runs = [b for b in with_def if check(name, b.strip()) is None]
    if runs or with_def:
        return (runs or with_def)[-1].strip()
    i = text.find(f"def {name}")
    return text[i:].strip() if i != -1 else ""


def state(body) -> str:
    me = body.observe()["self"]
    return f"you at ({me['x']},{me['y']}), hp {me['hp']}, carrying {me['inventory'] or 'nothing'}"


def passes(body, test) -> bool:
    if "placed" in test:
        return body.placed_near(test["placed"])
    return not missing(test, body.observe())


def trial(name, src, args, test, library, seeds=None):
    """Run the skill in each world; [(seed, ok, run, state)]."""
    out = []
    for s in seeds or SEEDS:
        body = SimBody(s)
        run = Sandbox(body.do, body.observe, max_actions=300, max_s=60).run(name, src, args, library)
        out.append((s, passes(body, test), run, state(body)))
        body.close()
    return out


def ask(url, msgs, max_tokens=1800) -> tuple[str, float]:
    t0 = time.monotonic()
    r = httpx.post(url, timeout=600, json={"model": "default_model", "messages": msgs, "max_tokens": max_tokens,
                                           "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or "", time.monotonic() - t0


def forge(url, lib: Library, tries=4, only=None, author="forge", log=print, redo=False):
    rows = []
    for name, args, what, test in TASKS:
        if (only and name not in only) or (name in lib.index and not redo):
            continue                                  # (a skill the library has already: kept)
        have = lib.sources()
        listing = "\n".join(f"  {n}{tuple(lib.index[n]['call'].split(',')) if lib.index[n]['call'] else '()'}: "
                            f"{lib.index[n]['doc']}" for n in have) or "  (none yet)"
        sig = f"{name}({', '.join('n' for _ in args)})"
        msgs = [{"role": "system", "content": API.replace("{recipes}", recipes_text())},
                {"role": "user", "content": f"Library skills you may call:\n{listing}\nWrite {sig}: {what}. It will be "
                                            f"called as {name}{args!r} by a player who starts carrying nothing, somewhere "
                                            f"in the world. It passes when this holds at the end: {json.dumps(test)}."}]
        row = {"skill": name, "attempts": []}
        for k in range(tries):
            text, gen_s = ask(url, msgs)
            src = code_of(text, name)
            res = trial(name, src, args, test, have)
            ok = sum(r[1] for r in res)
            row["attempts"].append({"gen_s": round(gen_s, 1), "pass": f"{ok}/{len(res)}", "src": src,
                                    "errors": [r[2]["error"] for r in res]})
            log(f"{name} try {k + 1}: {ok}/{len(res)} worlds, gen {gen_s:.0f}s, errors {[r[2]['error'] for r in res]}")
            if ok >= PASS:
                lib.add(name, src, ",".join("n" for _ in args), test, author, f"{ok}/{len(res)}", gen_s)
                break
            bad = next(r for r in res if not r[1])
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"It passed in {ok} of {len(res)} worlds. In world {bad[0]}: "
                      + (f"error: {bad[2]['error']}. " if bad[2]["error"] else f"it returned {bad[2]['value']!r}. ")
                      + "Last actions:\n" + "\n".join(bad[2]["trace"][-12:]) + f"\nAt the end: {state_from(bad)}. "
                      f"Test: {json.dumps(test)}. Reason in a ```text block about the cause, then write the fixed function."}]
        row["ok"] = any(int(a["pass"].split("/")[0]) >= PASS for a in row["attempts"])
        rows.append(row)
        old = json.loads(lib.path.joinpath("forge_log.json").read_text()) if lib.path.joinpath("forge_log.json").exists() else []
        lib.path.joinpath("forge_log.json").write_text(json.dumps([r for r in old if r["skill"] != name] + [row], indent=1))
    return rows


def state_from(r):
    return r[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-url", default="http://127.0.0.1:8091/v1/chat/completions")
    p.add_argument("--lib", default="agents/_skills")
    p.add_argument("--tries", type=int, default=4)
    p.add_argument("--only", help="comma list of skills")
    p.add_argument("--author", default="Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--redo", action="store_true", help="write again the skills the library has")
    a = p.parse_args()
    rows = forge(a.model_url, Library(Path(a.lib)), a.tries, a.only.split(",") if a.only else None, a.author,
                 log=lambda s: print(s, flush=True), redo=a.redo)
    print(json.dumps({"passed": [r["skill"] for r in rows if r["ok"]], "failed": [r["skill"] for r in rows if not r["ok"]]}))


if __name__ == "__main__":
    main()
