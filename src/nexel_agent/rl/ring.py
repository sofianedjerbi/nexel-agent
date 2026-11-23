"""The `ring` verb: walls placed from the pack on the 8 tiles round the player that are still open (a door on one, if
it carries one), one place after another, as the game's own place does it. The policy chooses when; it can be done
only when the pack holds enough to close the ring, every open tile can take a piece (bare ground), and nobody else
stands on or beside those tiles (it would wall them in or build where they walk)."""

from megastructure.sim.constants import VIEW_R

WALLS = ("stone_wall", "wood_wall")          # (the pieces that close a room: the game's ENCLOSE)
DOORS = ("door", "gate")
RING = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))


def plan(obs: dict, ground_ch, block_name) -> list | None:
    """[(dx, dy, item)] to close the ring round the player, or None when it cannot be done now."""
    grid, inv = obs.get("grid") or [], (obs.get("self") or {}).get("inventory") or {}
    if len(grid) <= VIEW_R + 1:
        return None
    for e in obs.get("entities", []):
        if e.get("kind") == "agent" and max(abs(e["dx"]), abs(e["dy"])) <= 2:
            return None                                  # (someone on or beside the ring)
    open_tiles = []
    for dx, dy in RING:
        ch = grid[VIEW_R + dy][VIEW_R + dx]
        name = block_name.get(ch)
        if name in WALLS or name in DOORS:
            continue                                     # (a piece already there)
        if ch not in ground_ch or ch == "~":
            return None                                  # (a tree, a rock, a bench, water...: not its wall)
        open_tiles.append((dx, dy))
    if not open_tiles:
        return None
    door = next((d for d in DOORS if inv.get(d)), None)
    walls = [w for w in WALLS for _ in range(inv.get(w, 0))]
    need = len(open_tiles) - (1 if door else 0)
    if len(walls) < need:
        return None
    side = [t for t in open_tiles if 0 in t]              # (a door goes on a side, not a corner)
    at = side[-1] if door and side else None
    if door and at is None:
        need += 1
        if len(walls) < need:
            return None
    out, k = [], 0
    for t in open_tiles:
        if t == at:
            continue
        out.append((*t, walls[k]))
        k += 1
    return out + ([(*at, door)] if at else [])
