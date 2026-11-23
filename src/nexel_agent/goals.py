"""A goal's own success test, written by the model with its goal (design 3.3), checked by the body after every
action, no model call: {"have": {"workbench": 1}} | {"placed": "workbench"} | {"near": "workbench"} |
{"at": [x, y]} | {"hp_at_least": 15} | {"food_at_least": 12}. Several keys: all must hold ("placed": one stands within
3 tiles). It says what is missing, as facts.
"""

from .verbs import locate

KEYS = ("have", "placed", "near", "at", "hp_at_least", "food_at_least")


def clean(test) -> dict | None:
    if not isinstance(test, dict):
        return None
    t = {k: v for k, v in test.items() if k in KEYS}
    if isinstance(t.get("have"), dict):
        t["have"] = {str(k): int(v) for k, v in t["have"].items() if str(v).lstrip("-").isdigit()}
    elif "have" in t:
        del t["have"]
    return t or None


def missing(test, obs, record=None, since: dict | None = None) -> list[str]:
    """What the test still lacks ([] = done). since: the placed counts when the goal was set."""
    me = obs["self"]
    inv = me["inventory"]
    out = []
    for k, n in (test.get("have") or {}).items():
        if inv.get(k, 0) < n:
            out.append(f"have {n} {k} (you have {inv.get(k, 0)})")
    if p := test.get("placed"):                       # (a state: one placed within 3 tiles, by anyone, at any time)
        if not any(max(abs(x - me["x"]), abs(y - me["y"])) <= 3 for x, y in locate(obs, str(p))):
            out.append(f"a {p} placed within 3 tiles")
    if n := test.get("near"):
        if not any(abs(x - me["x"]) <= 2 and abs(y - me["y"]) <= 2 for x, y in locate(obs, str(n))):
            out.append(f"within 2 tiles of a {n}")
    if (at := test.get("at")) and isinstance(at, list) and len(at) == 2:
        if abs(me["x"] - int(at[0])) + abs(me["y"] - int(at[1])) > 1:
            out.append(f"at ({at[0]},{at[1]}) (you are at ({me['x']},{me['y']}))")
    if (h := test.get("hp_at_least")) is not None and me["hp"] < float(h):
        out.append(f"health {h}+ (you have {me['hp']})")
    if (f := test.get("food_at_least")) is not None and me["hunger"] < float(f):
        out.append(f"food {f}+ (you have {me['hunger']})")
    return out
