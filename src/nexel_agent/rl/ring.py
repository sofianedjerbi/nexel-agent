"""The `ring` verb: the pieces in the pack placed on the open bare tiles of the 8 round the player (a door on a side,
if it carries one), one place after another, as the game's own place does it. As many as it holds, at least `least`;
the policy chooses when. Not while anyone else is on or beside those tiles (it would wall them in)."""

from megastructure.sim.constants import VIEW_R

WALLS = ("stone_wall", "wood_wall")          # (the pieces that close a room: the game's ENCLOSE)
DOORS = ("door", "gate")
RING = ((0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1))     # (the sides first)


def plan(obs: dict, ground_ch, least: int = 2) -> list | None:
    """[(dx, dy, item)] for the open tiles round the player, or None (fewer than `least` pieces can go down now)."""
    grid, inv = obs.get("grid") or [], (obs.get("self") or {}).get("inventory") or {}
    if len(grid) <= VIEW_R + 1 or any(e.get("kind") == "agent" and max(abs(e["dx"]), abs(e["dy"])) <= 2
                                      for e in obs.get("entities", [])):
        return None
    tiles = [(dx, dy) for dx, dy in RING if grid[VIEW_R + dy][VIEW_R + dx] in ground_ch - {"~"}]
    door = [next(d for d in DOORS if inv.get(d))] if any(inv.get(d) for d in DOORS) and tiles and 0 in tiles[0] else []
    pieces = door + [w for w in WALLS for _ in range(inv.get(w, 0))]
    out = list(zip(tiles, pieces))
    return [(dx, dy, it) for (dx, dy), it in out] if len(out) >= least else None
