"""A home's floor (the owner, 2026-11-28: "agent houses are missing a floor, they keep the default floor"): the
ground storey's inside laid over, as a player floors a house - planks laid on the ground are a wooden floor (the
game's "floor" ground), a path laid is cobble, a stone floor. Laid after its walls and door, before its furniture,
on every inside tile the furniture leaves bare (a bed stands on its own tile: no floor seen under it). A yard, a
garden and a home dug into the land have none (rl/structure.py cells)."""

FLOORS = {"floor": "plank", "stone_floor": "path"}                # the piece, and the item that lays it
NAMES = {"plank": "floor", "wood": "floor", "path": "stone_floor", "stone": "stone_floor", "cobble": "stone_floor"}
TOLD = {"floor": "planks", "stone_floor": "stone"}


def floor_of(spec) -> str | None:
    """The floor chosen: its piece ("floor": planks, the default; "stone_floor": cobble), or None (false: the land)."""
    v = spec.get("floor", "plank")
    if v in (False, None, "none"):
        return None
    if v not in NAMES:
        raise ValueError(f'a floor is "plank" (wood), "path" (stone) or false (the land), not {v!r}')
    return NAMES[v]


def with_floor(s, ground: list) -> list:
    """The ground storey's pieces with its floor laid after the walls and door: [(dx, dy, 0, piece)]."""
    piece = floor_of(s.spec) if s.roofed else None
    if piece is None:
        return ground
    things = s.storeys[0]
    floor = [(x, y, 0, piece) for x, y in sorted(s.inside, key=lambda p: (p[1], p[0])) if (x, y) not in things]
    shell = [c for c in ground if c[3] in ("wall", "door")]
    return shell + floor + [c for c in ground if c[3] not in ("wall", "door")]
