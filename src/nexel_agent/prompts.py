"""What the thinker tells the model: the rules of the world, the tools, recipes, and the observation as text."""

import json

from .sim import TOOL_SPECS
from .tiles import RECIPES

RULES = """You are a person living on an endless archipelago of wild green islands.
Nobody rules here. There are no laws and no safe zones: anything can be built, dug, claimed, traded,
stolen or destroyed. Others like you live here too; they may help you, trade with you, or rob you.

World facts:
- Top-down tile grid. North = up (y decreases). Coordinates are absolute (x, y).
- Hills: `heights` (when there are any) gives each tile's level (see Layers). move_to finds the way.
- Health is 0 to 20 (ten hearts, in halves), food (`hunger`) 0 to 20 (20 = full, 0 = starving). Food does not go
  down by itself: walking uses a little, sprinting three times that, fighting and digging some, healing a lot. With
  food at 18 or more you heal; at 6 or less you cannot sprint; at 0 you lose health. Eat berries, mushrooms, eggs,
  milk, cooked meat or bread (cooked food lasts longest).
- Dig adjacent things to harvest them: trees -> wood, berry bushes -> berries + seeds (cut the bare bush -> fiber),
  rocks -> stone (slow by hand), ore rocks -> ore (needs a pickaxe), mushrooms -> mushroom.
- A block of the land you dig out (dig with z, dig_down, a tunnel) stays a block in your pack: dirt (earth) or
  stone_block. place puts it back as land: on open ground (no z), or with x, y, z into any empty cell within reach
  that has something solid beside it (a wall, a roof, a step). For the material: craft stone = 1 stone_block (gives 2).
  What you built (walls, planks, roof, path) comes back as itself when you dig it.
- Layers: the land is blocks, in layers, from 512 under the sea to 512 over it (heights: 0-9, A-C for 10-12 and
  on to Z, 35 or higher; a, b, c... for -1, -2, -3 below sea level, z is 26 down or deeper; your own height, in
  full: z in your stats). A step one layer up stops step and walk:
  jump dir=X hops onto it (move_to jumps by itself); higher is a wall. Stepping off an edge you fall to the
  ground below: up to 3 layers is nothing, each layer more costs 5 hp (move_to never takes a fall that hurts).
  With a pickaxe in hand, dig toward a wall takes out the block in front of you (a step cut down, or a tunnel into
  the hill), and dig_down the block under your feet (by hand too, slower; you come down with it; into a hollow, you fall in).
  Up and out of any hole: dig dir=up (the block over your head), dig the block in front at z = your z + 2, jump; a layer each time. Caves are
  just empty space under the hills (in_cave in your stats): walk in where they open onto the ground.
- Items lying next to you come to you. Mining ore, crafting and hunting earn experience (xp, level). What an animal leaves when it dies is by chance: often meat, sometimes nothing, rarely something else.
- Digging bare ground tills it. Plant seeds in tilled soil; after a few minutes they ripen into wheat.
- A workbench (4 wood) unlocks tools: axe (faster trees), pickaxe (rock, ore), spear (weapon).
- Chickens and cows wander around; they are food.
- Creatures are rare. A tusker (boar) paws the ground, then charges in a straight line: step aside. A pincer (crab) on a shore
  snaps if you stand next to it. A strawman (scarecrow) walks toward you at night and fears firelight.
- You can swim on any water at walking speed (a boat is faster). Water has depth: ~ shallow, ( deeper, = deep, ) deepest.
  Digging at water takes its bottom a layer down and gives dirt; dirt put in water brings it back up.
- Old Mossback is a giant tortoise that sleeps as a hill with a tree on it. Hit it and it wakes: far too strong to fight alone.
- Death drops everything you carry where you fell. You come back, empty-handed, somewhere else.
- Everyone sees what you say nearby; shouting carries far but is tiring.
"""


FORMAT = """Reply with ONE JSON object and nothing else:
{"thought": "<one or two short sentences: what you see and why this action>",
 "tool": "<tool name>", "args": {...},
 "note": "<optional: a line to add to your notepad (plans, people, places). Omit if nothing new>"}"""


def tools_text() -> str:
    lines = []
    for name, spec in TOOL_SPECS.items():
        args = ", ".join(f"{k}: {v}" for k, v in spec["args"].items()) or "-"
        lines.append(f"- {name}({args}): {spec['desc']}")
    return "\n".join(lines)


def recipes_text() -> str:
    out = []
    for name, r in RECIPES.items():
        ins = " + ".join(f"{n} {k}" for k, n in r.inputs.items())
        out.append(f"{name} = {ins}" + (f" [near {r.station}]" if r.station else ""))
    return "; ".join(out)


def render_obs(obs: dict) -> str:
    me = obs["self"]
    grid = obs["grid"]
    r = len(grid) // 2
    # column/row offsets help the model convert grid cells to absolute coordinates
    header = "    " + "".join(str((me["x"] - r + i) % 10) for i in range(len(grid[0])))
    rows = [f"{me['y'] - r + j:>4}" + row for j, row in enumerate(grid)]
    lines = [f"Time: {obs['time']}. You are '@' at ({me['x']},{me['y']}), standing on {me['standing_on']}"
             + (", sheltered" if me["sheltered"] else "") + ".",
             f"Health {me['hp']}/20, food {me['hunger']}/20 (low = starving), "
             f"deaths so far {me['deaths']}.",
             "Inventory: " + (", ".join(f"{k} x{v}" for k, v in me["inventory"].items()) or "nothing") + ".",
             "Map (x digits on top are the last digit of x; y on the left):", header, *rows,
             "Legend: " + ", ".join(f"{k}={v}" for k, v in obs["legend"].items())]
    adj = []
    for d, (dx, dy) in {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items():
        adj.append(f"{d}={obs['legend'].get(grid[r + dy][r + dx], grid[r + dy][r + dx])}")
    lines.append("Adjacent: " + ", ".join(adj))
    if obs["entities"]:
        ents = []
        for e in obs["entities"][:10]:
            pos = f"({me['x'] + e['dx']},{me['y'] + e['dy']})"
            if e["kind"] == "agent":
                ents.append(f"{e['name']} [id {e['id']}] at {pos}, hp {e['hp']}, holding {e['holding']}, {e['doing']}")
            else:
                ents.append(f"{e['kind']} [id {e['id']}] at {pos}, hp {e['hp']}")
        lines.append("Nearby: " + "; ".join(ents))
    if obs["items"]:
        lines.append("Items on the ground: " + "; ".join(
            f"({me['x'] + i['dx']},{me['y'] + i['dy']}): " + ", ".join(f"{k} x{v}" for k, v in i["items"].items())
            for i in obs["items"][:6]))
    if obs["signs"]:
        lines.append("Signs: " + "; ".join(f"({me['x'] + s['dx']},{me['y'] + s['dy']}) {s['kind']}: \"{s['text']}\""
                                           for s in obs["signs"][:5]))
    heard = [e for e in obs["events"] if e["type"] in ("say", "shout", "give", "attack", "death", "sign")]
    if heard:
        lines.append("Recently: " + "; ".join(describe_event(e) for e in heard[-8:]))
    return "\n".join(lines)


def describe_event(e: dict) -> str:
    t = e["type"]
    if t in ("say", "shout"):
        return f"{e['from']} {'shouted' if t == 'shout' else 'said'} \"{e['text']}\""
    if t == "give":
        return f"{e['from']} gave {e['to']} {', '.join(f'{v} {k}' for k, v in e['items'].items())}"
    if t == "attack":
        return f"{e['attacker']} hit {e['target']} for {e['damage']}"
    if t == "death":
        return f"{e['name']} died ({e['cause']})"
    if t == "sign":
        return f"{e['by']} wrote \"{e['text']}\""
    return t


def summarize_result(tool, args, reply) -> str:
    a = json.dumps(args, separators=(",", ":"))
    if reply.get("ok"):
        res = reply["result"]
        ch = res.get("changed", {})
        bits = []
        if "inventory" in ch:
            bits.append(", ".join(f"{'+' if v > 0 else ''}{v} {k}" for k, v in ch["inventory"].items()))
        if "pos" in ch:
            bits.append(f"now at ({ch['pos'][0]},{ch['pos'][1]})")
        for k in ("text", "made", "target", "killed", "chest", "rows_near_to_far"):
            if k in res:
                bits.append(f"{k}={json.dumps(res[k])[:300]}")
        return f"{tool}{a} -> ok ({res['ticks']} ticks){': ' + '; '.join(bits) if bits else ''}"
    return f"{tool}{a} -> FAILED {reply.get('error')}: {reply.get('detail')}"
