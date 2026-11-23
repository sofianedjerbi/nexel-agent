"""Progress watchdog: if over WINDOW_S the body moved less than MIN_TILES net and the pack did not change, the model
is told that what it is doing gets nowhere here. It only tells: what to do instead is the model's choice.
"""

from collections import deque

WINDOW_S = 20.0
MIN_TILES = 3


class Watchdog:
    def __init__(self, window=WINDOW_S):
        self.window = window
        self.samples: deque = deque()            # (t, x, y, pack)
        self.stalls: list[dict] = []

    def record(self, t, x, y, pack: dict, routine: str = "you") -> dict | None:
        """After each look. Returns the stall ({routine, at, secs}) when one is declared, else None."""
        self.samples.append((t, x, y, tuple(sorted(pack.items()))))
        while len(self.samples) > 1 and t - self.samples[1][0] >= self.window:
            self.samples.popleft()                # (keeps the newest sample at least a window old)
        first = self.samples[0]
        if t - first[0] < self.window:
            return None
        if abs(x - first[1]) + abs(y - first[2]) >= MIN_TILES or any(s[3] != first[3] for s in self.samples):
            return None
        stall = {"routine": routine, "at": (x, y), "secs": round(t - first[0]), "t": t}
        self.stalls.append(stall)
        self.samples.clear()
        return stall


def stall_line(stall) -> str:
    x, y = stall["at"]
    return (f"(for {stall['secs']} s you have stayed around ({x},{y}) and nothing in your pack changed: what you are "
            "doing gets nowhere here. Do something different, or go somewhere else)")
