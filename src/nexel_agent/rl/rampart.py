"""A wall round a whole site (the owner: "a huge base complex"): the parts of a compound (rl/compound.py, "walled":
the gate's side) closed in by one ring of walls, open ground between them and it, a gate (a door) in the middle of
that side and a path out of it. Laid after every part (the builder walks in and out freely till then); judged on
its own: every tile of the ring walled, the gate a door. A home's rules (6 to 40 tiles inside, a bed) are not its."""

from .judge import ENCLOSE
from .structure import SIDE, Structure, door_at

MARGIN = 2                  # open ground between the parts (their paths counted) and the wall, a walk round inside


class Rampart(Structure):
    encloses = True

    def __init__(self, box, side="south"):           # (no footprint of its own: the site's box, not a home's size)
        side = str(side if side in SIDE else "south").lower()
        x0, y0, x1, y1 = box
        self.spec, self.shape, self.n, self.roof, self.roofed, self.wings = {"door": side}, "rect", 1, None, False, []
        self.inside = {(x, y) for x in range(x0 - MARGIN, x1 + MARGIN + 1) for y in range(y0 - MARGIN, y1 + MARGIN + 1)}
        X0, Y0, X1, Y1 = x0 - MARGIN - 1, y0 - MARGIN - 1, x1 + MARGIN + 1, y1 + MARGIN + 1
        self.ring = {(x, y) for x in range(X0, X1 + 1) for y in (Y0, Y1)} | {(x, y) for y in range(Y0, Y1 + 1) for x in (X0, X1)}
        self.doors, self.sides, self.stairs = [door_at(self.inside, self.ring, side)], [side], []
        self.storeys = [{**{p: "wall" for p in self.ring}, self.doors[0]: "door"}]
        self.path = self._line(self.doors[0], SIDE[side], 2)

    def cells(self) -> list:
        shell = sorted(self.storeys[0].items(), key=lambda kv: (kv[1] != "wall", kv[0][1], kv[0][0]))
        return [(x, y, 0, p) for (x, y), p in shell] + [(x, y, 0, "path") for x, y in self.path]

    def stands(self) -> tuple:
        """Inside, the tile in from the gate; the path; spare, round the ring outside."""
        g, (dx, dy) = self.doors[0], SIDE[self.sides[0]]
        out = {(x + i, y + j) for x, y in self.ring for i in (-1, 0, 1) for j in (-1, 0, 1)} - self.ring - self.inside
        return ((g[0] - dx, g[1] - dy, 0), *((x, y, 0) for x, y in self.path)), \
            tuple((x, y, 0) for x, y in sorted(out) if (x, y) not in self.path)

    def clear(self) -> set:
        return self.ring | set(self.path)

    def cost(self) -> int:
        return len(self.ring) - 1


def rampart_at(sim, s: Rampart, x0, y0) -> dict:
    """The wall as it stands in `sim` at (x0, y0): closed (every ring tile a wall or the gate) and its gate a door."""
    blocks = {p: sim.world.block(x0 + p[0], y0 + p[1]) for p in s.ring}
    gaps = sum(b not in ENCLOSE for b in blocks.values())
    failed = (["closed"] if gaps else []) + ([] if blocks[s.doors[0]] in ("door", "gate") else ["gate"])
    return {"ok": True, "grade": 2 - len(failed), "of": 2, "failed": failed, "kind": "wall",
            "told": "; ".join({"closed": f"the wall has {gaps} gaps", "gate": "no gate (a door) in the wall"}[f]
                              for f in failed) or "nothing failed"}
