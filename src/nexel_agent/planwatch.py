"""Plans that change nothing: the model asked the same plan again and the pack is just as it was.

A repeat is told back to the model (it decides what to do instead): nothing is refused or replaced.
"""

from collections import Counter

TOLD, REFUSED = 1, 2


class PlanWatch:
    def __init__(self):
        self.seen: dict[tuple, dict] = {}         # plan -> the pack when it last arrived
        self.nothing: Counter = Counter()          # plan -> times in a row it changed nothing
        self.pack: dict | None = None

    def arrived(self, plan: tuple, inv: dict) -> int:
        """Record a finished plan; how many times in a row the same plan changed nothing before it."""
        inv = dict(inv)
        if self.pack is not None and inv != self.pack:    # something changed: every plan gets a fresh chance
            self.nothing.clear()
            self.seen.clear()
        self.pack = inv
        if self.seen.get(plan) == inv:
            self.nothing[plan] += 1
        self.seen[plan] = inv
        return self.nothing[plan]

    def refused(self, first: str) -> tuple | None:
        """A plan known to change nothing that starts with this command, if any."""
        return next((p for p, n in self.nothing.items() if n >= REFUSED and p and p[0] == first), None)


def did_nothing(plan: tuple, why: str) -> str:
    return (f"(your plan '{', '.join(plan)}' changed nothing{': ' + why if why else ''}. "
            "Only what Near lists is really there: pick something from it)")
