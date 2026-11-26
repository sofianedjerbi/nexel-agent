"""Goals set aside: the planner may change its goal whenever it sees fit, and what it was after is not forgotten.
A goal replaced before it was done goes on the agenda, with how far it got; the planner sees the agenda in its facts
and sets one again to go back to it, or lets it go. Facts only: nothing here says which to take up, or when. Kept in
memory.json across sessions, as the aims are. (Live: invited to fight a boss, Rook could only drop its fiber ask
or ignore the invitation; with an agenda it can go, and come back to the fiber after.)"""

import time

KEEP_S = 20 * 60            # an entry is let go after this
MOST = 4                    # entries kept, the newest


class Agenda:
    def __init__(self, saved=()):
        self.items = [dict(e) for e in saved if isinstance(e, dict) and e.get("kind")]

    @staticmethod
    def key(e) -> tuple:
        return e.get("kind"), e.get("item"), e.get("person")

    def set_aside(self, g, how: str):
        """The goal `g`, replaced before it was done: kept, with how far it got (goals.account)."""
        if g.kind == "free":
            return
        e = {"kind": g.kind, "item": g.item, "person": g.person, "at": round(time.time()), "how": how}
        self.items = [x for x in self.items if self.key(x) != self.key(e)][-(MOST - 1):] + [e]

    def taken_up(self, g):
        """Set again, or done: off the agenda."""
        self.items = [x for x in self.items if self.key(x) != (g.kind, g.item, g.person)]

    def line(self) -> str | None:
        """The facts line, or None when there is nothing set aside."""
        now = time.time()
        self.items = [x for x in self.items if now - x["at"] <= KEEP_S]
        if not self.items:
            return None
        told = [f"{self.said(x)} ({(now - x['at']) / 60:.0f} min ago; {x['how']})" for x in reversed(self.items)]
        return "Goals you set aside, yours to go back to or let go: " + "; ".join(told) + "."

    @staticmethod
    def said(x) -> str:
        who = f"(with {x['person']})" if x.get("person") else None
        return " ".join(str(p) for p in (x["kind"], x.get("item"), who) if p)

    def saved(self) -> list:
        return list(self.items)
