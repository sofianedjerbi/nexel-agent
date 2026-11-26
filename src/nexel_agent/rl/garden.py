"""A garden (#66): a plot fenced round with a gate in the middle of a side, beds of tilled soil inside sown with seed,
walks between the beds, a path out of the gate - as the game farms (nexel_worlds/sim/building.py: `dig` with a hoe
tills bare ground; a seed takes on tilled soil only and grows; water from a can, twice as fast).

    = = G = =          the fence (f), the gate (G), beds (o, sown) in rows along the long side, a walk (.) between
    f o o o f          each pair of beds, so each bed is reached from a walk or the fence's side
    f . . . f

A design as the architect chooses it: {"garden": true, "shape", "width"/"depth" or "radius", "door" (the gate's
side), "beds": "rows" (a walk between each two, the default) or "full" (all soil, walked on), "path"}."""

from .structure import SIDE, Structure


class Garden(Structure):
    def __init__(self, spec: dict):
        super().__init__({**spec, "storeys": 1, "roof": False, "furniture": {}})
        self.beds_kind = spec.get("beds", "rows")
        if self.beds_kind not in ("rows", "full"):
            raise ValueError('beds are "rows" or "full"')
        gate = self.doors[0]
        self.storeys = [{**{p: "fence" for p in self.ring}, gate: "gate"}]
        xs, ys = [x for x, _ in self.inside], [y for _, y in self.inside]
        along_x = max(xs) - min(xs) >= max(ys) - min(ys)
        front = next(q for q in ((gate[0] + d[0], gate[1] + d[1]) for d in SIDE.values()) if q in self.inside)
        line = (lambda p: p[1] - min(ys)) if along_x else (lambda p: p[0] - min(xs))      # (the rows along the long side)
        walk = (lambda p: line(p) % 3 == 1) if self.beds_kind == "rows" else (lambda p: False)
        spine = (lambda p: p[0] == front[0]) if SIDE[self.sides[0]][0] == 0 else (lambda p: p[1] == front[1])
        self.beds = sorted(p for p in self.inside if not walk(p) and not (self.beds_kind == "rows" and spine(p)))
        self.walks = sorted(set(self.inside) - set(self.beds))      # (from the gate, a spine across the rows)

    def cells(self) -> list:
        """The fence and its gate, the path, then each bed tilled and sown."""
        shell = sorted(self.storeys[0].items(), key=lambda kv: (kv[1] != "fence", kv[0][1], kv[0][0]))
        return ([(x, y, 0, p) for (x, y), p in shell] + [(x, y, 0, "path") for x, y in self.path]
                + [(x, y, 0, "till") for x, y in self.beds] + [(x, y, 0, "seed") for x, y in self.beds])

    def stands(self) -> tuple:
        """The walks (in rows: every bed is beside one or the fence), the path; spare, round the fence outside."""
        st = [(x, y, 0) for x, y in self.walks] + [(x, y, 0) for x, y in self.path]
        out = {(x + i, y + j) for x, y in self.ring for i in (-1, 0, 1) for j in (-1, 0, 1)} - self.ring - self.inside
        return tuple(st), tuple((x, y, 0) for x, y in sorted(out) if (x, y) not in self.path)

    def clear(self) -> set:
        return self.inside | self.ring | set(self.path)

    def cost(self) -> int:
        return len(self.ring)                       # (the fence: what is gathered and made; seeds are found)
