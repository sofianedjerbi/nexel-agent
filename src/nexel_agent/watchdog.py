"""Progress watchdog over everything the body does (plans, filler, escape): three live faults in a row were one
routine repeating forever without getting anywhere (a plan for a tree that was not there, a filler walking into
cliffs, an escape stepping between two tiles). Whatever the source: if over WINDOW_S the body moved less than
MIN_TILES net and the pack did not change, the routine that ran most in that window has failed here. It is
banned in this area for BAN_S, and the next one is tried (and the model is told).
"""

from collections import Counter, deque

WINDOW_S = 20.0
MIN_TILES = 3
BAN_S = 180.0
AREA = 8                                         # a ban holds for this many tiles around (an area of the map)


def area(x, y) -> tuple:
    return x // AREA, y // AREA


class Watchdog:
    def __init__(self, window=WINDOW_S):
        self.window = window
        self.samples: deque = deque()            # (t, x, y, pack, routine)
        self.bans: dict[tuple, float] = {}       # (routine, area) -> until
        self.stalls: list[dict] = []

    def record(self, t, x, y, pack: dict, routine: str) -> dict | None:
        """After each action. Returns the stall ({routine, at, secs}) when one is declared, else None."""
        self.samples.append((t, x, y, tuple(sorted(pack.items())), routine))
        while len(self.samples) > 1 and t - self.samples[1][0] >= self.window:
            self.samples.popleft()                # (keeps the newest sample at least a window old)
        first = self.samples[0]
        if t - first[0] < self.window:
            return None
        moved = abs(x - first[1]) + abs(y - first[2])
        if moved >= MIN_TILES or any(s[3] != first[3] for s in self.samples):
            return None
        routine = Counter(s[4] for s in self.samples).most_common(1)[0][0]
        self.bans[(routine, area(x, y))] = t + BAN_S
        stall = {"routine": routine, "at": (x, y), "secs": round(t - first[0]), "t": t}
        self.stalls.append(stall)
        self.samples.clear()
        return stall

    def banned(self, x, y, t) -> set[str]:
        a = area(x, y)
        return {r for (r, ar), until in self.bans.items() if ar == a and until > t}


def stall_line(stall) -> str:
    x, y = stall["at"]
    return (f"(you have been going back and forth at ({x},{y}) for {stall['secs']} s with nothing gained, "
            f"doing '{stall['routine']}': that does not work here. Do something different, somewhere else)")
