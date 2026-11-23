"""A failed command, said back so it can be put right: the command, the error, and what would work instead (the right
argument form, what you carry that can be placed, the next missing step), computed from the recipe book and the view.
"""

from .ladder import RECIPES, _nearest, _walk, free_dirs, placeable_carried, station_steps, steps_for
from .plans import command_text
from .sim.constants import VIEW_R
from .sim.specs import LEGEND
from .tiles import FOOD, PLACEABLE

PLAIN = set("~.,:'=o_ ")


def _steps(name, inv, obs, k=3) -> str:
    return " > ".join(steps_for(name, dict(inv), obs)[:k])


GATHER = "TbMvHQF*"                   # what a dig gathers: tree, bushes, rock, mushroom, fruit tree, ore


def gather_hint(obs) -> str:
    """What is next to you to dig, or the nearest such thing and how to get at it."""
    g, r = obs["grid"], len(obs["grid"]) // 2
    side = [f"{LEGEND.get(g[r + dy][r + dx], '?').split(' (')[0]} {d}" for d, (dx, dy) in
            (("N", (0, -1)), ("E", (1, 0)), ("S", (0, 1)), ("W", (-1, 0))) if g[r + dy][r + dx] in GATHER]
    if side:
        return "Next to you to dig: " + ", ".join(side) + "."
    near = _nearest(obs, GATHER)
    if not near:
        return "Nothing to gather in view: explore."
    way = _walk(obs, near[1], near[2])
    return f"Nearest to gather: {LEGEND.get(near[3], '?').split(' (')[0]} at ({near[1]},{near[2]})" + (f": {way}." if way else ".")


def fix_hint(tool, args, error, detail, obs) -> str:
    """What to do about it, or '' (then the server's own words stand)."""
    inv = obs["self"]["inventory"]
    item = str(args.get("item") or args.get("recipe") or "")
    dirs = free_dirs(obs)
    if tool == "place":
        can = placeable_carried(inv)
        carried = f"you can place: {', '.join(can)}" if can else "you carry nothing placeable"
        if error == "not_placeable":
            made = [n for n, r in RECIPES.items() if item in r.inputs and n in PLACEABLE][:5]
            return (f"{item} is a material, not a thing to put down; {carried}"
                    + (f"; make something of it first: {', '.join(made)}" if made else "") + ".")
        if error == "missing_item":
            return f"you carry no {item}; {carried}" + (f"; to make one: {_steps(item, inv, obs)}" if item in RECIPES else "") + "."
        if error in ("cliff_edge", "occupied", "bad_target"):
            return (f"free flat sides now: {', '.join(dirs)} (e.g. place {dirs[0]} {item})" if dirs
                    else "no free flat side here: move_to open flat ground first") + "."
    if tool == "craft":
        rec = RECIPES.get(item)
        if error == "unknown_recipe" or rec is None:
            return f"no recipe '{item}'; the recipe names are in the list you were given (e.g. craft workbench)."
        if error == "need_station":
            st = station_steps(rec.station, dict(inv), obs)
            return f"{item} is made within 2 tiles of a placed {rec.station}: first " + " > ".join(st[:3]) + "."
        if error == "missing_items":
            have = ", ".join(f"{n} {k} (have {inv.get(k, 0)})" for k, n in rec.inputs.items())
            return f"{item} takes {have}; next: {_steps(item, inv, obs)}."
    if tool in ("dig", "dig_down") and error == "need_tool":
        need = next((t for t in ("hoe", "pickaxe", "axe") if t in (detail or "")), "pickaxe")
        if "tilling" in (detail or ""):
            till = f"dig {args.get('dir', 'N')} hoe" if inv.get("hoe") else "a hoe: " + _steps("hoe", inv, obs, 3)
            return (f"{args.get('dir', 'that way')} of you is bare ground: digging it tills it (for seeds). "
                    f"{gather_hint(obs)} To till: {till}.")
        if inv.get(need):
            return f"you carry a {need}: say it, e.g. dig {args.get('dir', 'N')} {need}."
        return f"you need a {need}: {_steps(need, inv, obs, 4)}."
    if tool == "attack" and error in ("no_target", "bad_args"):
        ids = [f"#{e['id']} {e.get('name') or e['kind']}" for e in obs["entities"][:4]]
        return "attack takes a being's id (the #number in Beings), e.g. attack 12; in view: " + (", ".join(ids) or "nobody") + "."
    if tool == "pick_up" and error == "nothing_here":
        return "pick_up only takes items lying on your own tile; to harvest a plant or tree: dig D toward it."
    if tool == "eat":
        if error == "full":
            return "you are full: eat later, do something else."
        food = [k for k in inv if k in FOOD]
        return f"food you carry: {', '.join(food) or 'none'}."
    return ""


def failure_line(tool, args, reply, obs) -> str:
    """'place N wood FAILED not_placeable: wood is a material...' (one line for the model's 'Done' list)."""
    err, detail = reply.get("error"), str(reply.get("detail") or "")
    hint = fix_hint(tool, args, err, detail, obs)
    return f"{command_text(tool, args)} FAILED {err}: {hint or detail[:120]}"[:260]


def look_far_line(args, res) -> str:
    """look_far said as what is out there and how far ('tree ~13 tiles, rock ~21 tiles'), not 24 rows of chars."""
    rows, d = res.get("rows_near_to_far") or [], args.get("dir", "?")
    seen: dict = {}
    for i, row in enumerate(rows):
        for ch in row:
            if ch not in PLAIN and ch not in seen:
                seen[ch] = VIEW_R + 1 + 2 * i
    things = [f"{LEGEND.get(c, c).split(' (')[0]} ~{n} tiles" for c, n in sorted(seen.items(), key=lambda kv: kv[1])[:6]]
    water = next((VIEW_R + 1 + 2 * i for i, row in enumerate(rows) if row.count("~") > len(row) // 2), None)
    people = [f"{x['name']} ({x['dx']:+d},{x['dy']:+d})" for x in res.get("agents") or []]
    none = [n for c, n in (("T", "tree"), ("M", "rock")) if c not in seen]
    return (f"look_far {d} -> " + (", ".join(things) or "only open ground") + (f"; no {' or '.join(none)}" if none else "") + (f"; water from ~{water} tiles" if water else "")
            + (f"; people: {', '.join(people)}" if people else "") + f" ({d} of you)")
