"""A failed command, said back as facts: what happened and the game's rule behind it (what can be placed, what a recipe
takes and where its parts are, what is next to you), computed from the recipe book and the view. Never what to do:
the model plans that.
"""

from .ladder import RECIPES, _nearest, free_dirs, placeable_carried, recipe_facts, where_to_stand
from .plans import command_text
from .sim.constants import VIEW_R
from .sim.specs import LEGEND
from .tiles import FOOD, PLACEABLE

PLAIN = set("~.,:'=o_ ")
GATHER = "TbMvHQF*"                   # what a dig gathers: tree, bushes, rock, mushroom, fruit tree, ore


def name_of(ch) -> str:
    return LEGEND.get(ch, "?").split(" (")[0]


def around(obs) -> str:
    """What is next to you that a dig gathers, or the nearest such thing and where to stand for it."""
    g, r = obs["grid"], len(obs["grid"]) // 2
    side = [f"{name_of(g[r + dy][r + dx])} {d}" for d, (dx, dy) in
            (("N", (0, -1)), ("E", (1, 0)), ("S", (0, 1)), ("W", (-1, 0))) if g[r + dy][r + dx] in GATHER]
    if side:
        return "Next to you: " + ", ".join(side) + "."
    near = _nearest(obs, GATHER)
    if not near:
        return "Nothing a dig gathers is in view."
    return f"Nearest thing a dig gathers: {name_of(near[3])} at ({near[1]},{near[2]}): {where_to_stand(obs, near[1], near[2])}."


def fix_hint(tool, args, error, detail, obs) -> str:
    """The facts behind the failure, or '' (then the server's own words stand)."""
    inv = obs["self"]["inventory"]
    item = str(args.get("item") or args.get("recipe") or "")
    dirs = free_dirs(obs)
    if tool in ("place", "place_near"):
        can = placeable_carried(inv)
        carried = f"placeable things you carry: {', '.join(can)}" if can else "you carry nothing placeable"
        if error == "not_placeable":
            made = [n for n, r in RECIPES.items() if item in r.inputs and n in PLACEABLE][:5]
            return (f"{item} is a material and cannot be placed; {carried}"
                    + (f"; placeable things made from {item}: {', '.join(made)}" if made else "") + ".")
        if error == "missing_item":
            return f"you carry no {item}; {carried}" + ("; " + "; ".join(recipe_facts(item, inv, obs)) if item in RECIPES else "") + "."
        if error in ("cliff_edge", "occupied", "bad_target", "no_room"):
            return f"{detail[:80]}; free flat sides next to you now: {', '.join(dirs) or 'none'}."
    if tool in ("craft", "craft_here"):
        rec = RECIPES.get(item.split()[0] if item else "")
        if error == "unknown_recipe" or rec is None:
            return f"there is no recipe '{item}'."
        if error in ("need_station", "missing_items"):
            return "; ".join(recipe_facts(rec.name, inv, obs)) + "."
    if tool in ("dig", "dig_down", "dig_at") and error == "need_tool":
        need = next((t for t in ("hoe", "pickaxe", "axe") if t in (detail or "")), "pickaxe")
        carry = f"you carry a {need}" if inv.get(need) else f"you carry no {need}"
        if "tilling" in (detail or ""):
            return (f"{args.get('dir', 'there')} of you is bare ground; digging bare ground tills it, which takes a hoe in "
                    f"hand (with hoe); {carry}. {around(obs)}")
        return f"{detail[:80]}; {carry}" + ("" if inv.get(need) else "; " + "; ".join(recipe_facts(need, inv, obs))) + "."
    if tool == "attack" and error in ("no_target", "bad_args"):
        ids = [f"#{e['id']} {e.get('name') or e['kind']}" for e in obs["entities"][:4]]
        return "attack takes a being's id (the #number in Beings); in view: " + (", ".join(ids) or "nobody") + "."
    if tool in ("dig_at", "go_next_to", "move_away", "where") and error == "not_found":
        far = (obs.get("_seen") or {}).get(str(args.get("text", "")).split(" with ")[0].strip().replace("_", " ")) or \
            (obs.get("_seen") or {}).get(str(args.get("text", "")).split(" with ")[0].strip())
        return f"{detail[:60]}" + (f"; you saw one far off with look_far at ({far[0]},{far[1]})" if far else "") + "."
    if tool == "pick_up" and error == "nothing_here":
        return f"pick_up takes items lying on your own tile; none lie there. {around(obs)}"
    if tool == "eat":
        if error == "full":
            return f"you are full (food {obs['self']['hunger']}/20)."
        food = [k for k in inv if k in FOOD]
        return f"food you carry: {', '.join(food) or 'none'}."
    return ""


def failure_line(tool, args, reply, obs) -> str:
    """'place N wood FAILED not_placeable: wood is a material...' (one line for the model's 'Done' list)."""
    err, detail = reply.get("error"), str(reply.get("detail") or "")
    hint = fix_hint(tool, args, err, detail, obs)
    return f"{command_text(tool, args)} FAILED {err}: {hint or detail[:120]}"[:260]


def look_far_line(args, res, me=None, seen=None) -> str:
    """look_far said as what is out there, how far and where ('tree ~13 tiles at (4,-13)'), not 24 rows of chars.
    seen (a dict) keeps the nearest place of each thing for later (what a failed dig_at tells)."""
    rows, d = res.get("rows_near_to_far") or [], str(args.get("dir", "?")).upper()
    fx, fy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.get(d, (0, 0))
    px, py = -fy, fx
    found: dict = {}
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch not in PLAIN and ch not in found:
                dist, side = VIEW_R + 1 + 2 * i, -8 + 2 * j
                at = (me["x"] + fx * dist + px * side, me["y"] + fy * dist + py * side) if me else None
                found[ch] = (dist, at)
    things = []
    for c, (n, at) in sorted(found.items(), key=lambda kv: kv[1][0])[:6]:
        things.append(f"{name_of(c)} ~{n} tiles" + (f" at ({at[0]},{at[1]})" if at else ""))
        if seen is not None and at:
            seen[name_of(c).replace("_", " ")] = at
    water = next((VIEW_R + 1 + 2 * i for i, row in enumerate(rows) if row.count("~") > len(row) // 2), None)
    people = [f"{x['name']} ({x['dx']:+d},{x['dy']:+d})" for x in res.get("agents") or []]
    none = [n for c, n in (("T", "tree"), ("M", "rock")) if c not in found]
    return (f"look_far {d} -> " + (", ".join(things) or "only open ground") + (f"; no {' or '.join(none)}" if none else "")
            + (f"; water from ~{water} tiles" if water else "") + (f"; people: {', '.join(people)}" if people else "")
            + f" ({d} of you)")
