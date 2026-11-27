"""Houses of wings (rl/structure.py, shape "wings"): rectangles joined side to side into one inside - an L, a T, a
U, a cross - each wing under its own gable, the ridges meeting where the wings do."""

from . import blueprint as bp


def wings(spec) -> list:
    """Each wing's inside tiles, from [[x, y, width, depth], ...] (from the inside's top-left), set from (1, 1); the
    wings joined into one inside, side to side."""
    ws = spec.get("wings")
    if not isinstance(ws, list) or not 2 <= len(ws) <= 4 or not all(isinstance(w, list) and len(w) == 4 for w in ws):
        raise ValueError("wings: 2 to 4 of [x, y, width, depth]")
    ws = [[int(v) for v in w] for w in ws]
    if any(not 2 <= w <= bp.MOST - 2 or not 2 <= d <= bp.MOST - 2 for _, _, w, d in ws):
        raise ValueError(f"a wing's width and depth from 2 to {bp.MOST - 2}")
    x0, y0 = min(w[0] for w in ws), min(w[1] for w in ws)
    out = [{(x - x0 + i + 1, y - y0 + j + 1) for i in range(w) for j in range(d)} for x, y, w, d in ws]
    allt = set().union(*out)
    if max(x for x, _ in allt) > bp.MOST - 2 or max(y for _, y in allt) > bp.MOST - 2:
        raise ValueError(f"the wings together at most {bp.MOST - 2} tiles each way")
    seen, todo = set(), [next(iter(allt))]
    while todo:
        p = todo.pop()
        if p in allt and p not in seen:
            seen.add(p)
            todo += [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]
    if seen != allt:
        raise ValueError("the wings must touch side to side: one inside")
    return out


def gable(ts, p) -> int:
    """How far up a gable over `ts` (a wing: its box with walls and eaves) is at p: rows in from the long sides."""
    xs, ys = [x for x, _ in ts], [y for _, y in ts]
    if not (min(xs) <= p[0] <= max(xs) and min(ys) <= p[1] <= max(ys)):
        return 0
    if max(xs) - min(xs) >= max(ys) - min(ys):                     # (the ridge along the long side)
        return int((max(ys) - min(ys)) / 2 - abs(p[1] - (min(ys) + max(ys)) / 2))
    return int((max(xs) - min(xs)) / 2 - abs(p[0] - (min(xs) + max(xs)) / 2))
