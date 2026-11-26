"""How an episode can start, besides at the spawn of a fresh world: somewhere else on open land, on land as players
leave it, or one step from a tool. Each is a state a player of the live game can be found in; none says what to do."""

import math

from nexel_worlds.tiles import RECIPES

TOOLS = ("spear", "axe", "pickaxe")


def somewhere(sim, a, rng):
    """Somewhere on open land within 400 tiles of the spawn (a live world's players are rarely where it began)."""
    for _ in range(40):
        x, y = a.x + rng.randint(-400, 400), a.y + rng.randint(-400, 400)
        if sim.stand(x, y) and sim.world.ground(x, y) in ("grass", "dirt"):
            a.x, a.y, a.z = x, y, 99
            return


def thin_land(sim, a, rng, r=40):
    """Land as players leave it near them (the live save): most trees round the start felled, and a few thickets
    and rocks spread out, so wood, fiber and stone take travelling for."""
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if sim.world.block(a.x + dx, a.y + dy) in ("tree", "fruit_tree") and rng.random() < 0.9:
                sim.world.set(a.x + dx, a.y + dy, block="none")
    for block, n in (("thicket", rng.randint(3, 6)), ("rock", rng.randint(2, 4)), ("tree", rng.randint(3, 8))):
        for _ in range(n * 10):
            d, ang = rng.uniform(12, r), rng.uniform(0, 6.283)
            x, y = a.x + round(d * math.cos(ang)), a.y + round(d * math.sin(ang))
            if sim.world.block(x, y) == "none" and sim.world.ground(x, y) in ("grass", "dirt"):
                sim.world.set(x, y, block=block)
                n -= 1
                if n <= 0:
                    break


def near_a_tool(sim, a, rng):
    """One step from a tool: most of one tool's materials in the pack (now and then one is lacking, or there is some to
    spare), and, half the time, a workbench on the tile beside it."""
    for item, n in RECIPES[rng.choice(TOOLS)].inputs.items():
        have = n + rng.choice((-1, 0, 0, 0, 1, 2))
        if have > 0:
            a.inventory[item] = a.inventory.get(item, 0) + have
    spots = [(a.x + dx, a.y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
             if sim.world.block(a.x + dx, a.y + dy) == "none" and sim.stand(a.x + dx, a.y + dy)]
    if spots and rng.random() < 0.5:
        sim.set_tile(*rng.choice(spots), block="workbench")


def with_a_home_kit(sim, a, rng):
    """The pieces of a home in the pack (a player who came back with them): the walls, a door, a bed, a fire, the
    roof (rl/build.py's hut), now and then short of the roof, the bed or the fire, so it makes or fetches the rest. Never
    of a wall or the door: r58 laid 28 pieces round a gap and left to make the door, and the room never closed."""
    from .build import PLAN
    kit = {}
    for _, _, piece in PLAN:
        item = rng.choice(("stone_wall", "wood_wall")) if piece == "wall" else piece
        kit[item] = kit.get(item, 0) + 1
    if rng.random() < 0.5:                               # (short of what a room stands without: the walls and the
        short = rng.choice(sorted(k for k in kit if k in ("roof", "bed", "campfire")))      # door close it)
        kit[short] = max(0, kit[short] - rng.randint(1, kit[short]))
    for k, n in kit.items():
        if n:
            a.add(k, n)
