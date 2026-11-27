"""Builds of several parts on one site (#69): a house with its garden beside it, a tower by a walled yard, a dug home
under a hill with beds at its door. Each part a design of its own (rl/structure.py, rl/dug.py, rl/garden.py), set
beside those before it on the side it names ("at": "east"...), a tile of open ground between them; laid a part after the
other, judged each as itself. A part added later (the mind names another build once the first stands) is the same
compound with one more part: what was laid reads as laid.

    {"parts": [{...the house...}, {"garden": true, "width": 5, "depth": 4, "at": "east"}]}"""

from .structure import SIDE

GAP = 2                     # tiles of open ground between two parts (a fence nearer a wall reads as a piece astray)
MOST_PARTS = 4


class Compound:
    def __init__(self, spec: dict):
        from .dug import design_of
        parts = spec.get("parts") or []
        if not 1 <= len(parts) <= MOST_PARTS:
            raise ValueError(f"parts: 1 to {MOST_PARTS} designs")
        if any(p.get("parts") for p in parts):
            raise ValueError("a part is one design, not parts of its own")
        self.spec, self.parts, self.at = spec, [design_of(p) for p in parts], []
        x0, y0, x1, y1 = _box(self.parts[0].clear())
        self.at.append((0, 0))
        for s, p in zip(self.parts[1:], parts[1:]):
            side = str(p.get("at", "east")).lower()
            if side not in SIDE:
                raise ValueError(f"a part's at is one of {', '.join(SIDE)}")
            a0, b0, a1, b1 = _box(s.clear())
            dx, dy = SIDE[side]
            ox = x1 + GAP + 1 - a0 if dx > 0 else x0 - GAP - 1 - a1 if dx < 0 else (x0 + x1) // 2 - (a0 + a1) // 2
            oy = y1 + GAP + 1 - b0 if dy > 0 else y0 - GAP - 1 - b1 if dy < 0 else (y0 + y1) // 2 - (b0 + b1) // 2
            self.at.append((ox, oy))
            x0, y0 = min(x0, a0 + ox), min(y0, b0 + oy)
            x1, y1 = max(x1, a1 + ox), max(y1, b1 + oy)
        self.n, self.inside = 1, self.parts[0].inside          # (what the mind's home is read from: the first part)

    def _moved(self, k, tiles):
        ox, oy = self.at[k]
        return {(x + ox, y + oy) for x, y in tiles}

    def _cells(self, k, cells):
        ox, oy = self.at[k]
        return [(c[0] + ox, c[1] + oy, *c[2:]) for c in cells]

    def cells(self) -> list:
        return [c for k, s in enumerate(self.parts) for c in self._cells(k, s.cells())]

    def stands(self) -> tuple:
        st = [self._cells(k, s.stands()[0]) for k, s in enumerate(self.parts)]
        sp = [self._cells(k, s.stands()[1]) for k, s in enumerate(self.parts)]
        taken = self.clear()
        return tuple(x for part in st for x in part), tuple(x for part in sp for x in part if x[:2] not in taken)

    def clear(self) -> set:
        return set().union(*(self._moved(k, s.clear()) for k, s in enumerate(self.parts)))

    def from_(self) -> dict:
        return {self._cells(k, [c])[0]: tuple(self._cells(k, list(v))) for k, s in enumerate(self.parts)
                for c, v in s.from_().items()}

    def ways(self) -> dict:
        return {self._cells(k, [st])[0]: tuple(self._cells(k, list(v))) for k, s in enumerate(self.parts)
                for st, v in s.ways().items()}

    @property
    def stairs(self) -> list:
        """Each part's flights of stairs, where the part is set: (top, steps, laid from, storey)."""
        mv = lambda k, ps: [(x + self.at[k][0], y + self.at[k][1]) for x, y in ps]        # noqa: E731
        return [(mv(k, [top])[0], mv(k, steps), tuple(mv(k, at)), n) for k, s in enumerate(self.parts)
                for top, steps, at, n in getattr(s, "stairs", [])]

    def part_of(self) -> dict:
        """Which part each cell is of (a roof waits for its own part's walls, not the garden's fence)."""
        return {c: k for k, s in enumerate(self.parts) for c in self._cells(k, s.cells())}

    def cost(self) -> int:
        return sum(s.cost() for s in self.parts)

    def grid(self, k=0) -> list:
        return self.parts[0].grid(k)


def _box(tiles):
    xs, ys = [x for x, _ in tiles], [y for _, y in tiles]
    return min(xs), min(ys), max(xs), max(ys)


def describe(spec: dict) -> str:
    """A design in a few words, for the mind's facts: "a two-storey house, a gable roof of shingles; a garden east"."""
    if spec.get("parts"):
        return "; ".join(describe(p) + (f" {p['at']}" if k and p.get("at") else "") for k, p in enumerate(spec["parts"]))
    shape = "round" if spec.get("shape") == "round" else ""
    if spec.get("garden"):
        return " ".join(w for w in ("a", shape, "garden") if w)
    if spec.get("underground"):
        return " ".join(w for w in ("a", shape, "home dug underground") if w)
    n = int(spec.get("storeys", 1))
    roof = spec.get("roof", True)
    what = " ".join(w for w in ("a", shape, "walled yard") if w) if roof is False else " ".join(
        w for w in ("a", {2: "two-storey", 3: "three-storey"}.get(n, ""), shape, "house" if n < 3 else "tower") if w)
    if isinstance(roof, dict):
        mat = {"roof": "shingles", "stone_block": "stone", "dirt": "earth"}.get(roof.get("material", "roof"),
                                                                                    roof.get("material", "roof"))
        what += f", a {roof.get('style', 'flat')} roof of {mat}"
    return what
