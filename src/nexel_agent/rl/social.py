"""Who spoke to the player, what they asked for, who gave it what: read from the observation's events, the same in
the sim and on the live server. The policy sees three flags of it (rl/encode.py); the social achievements (+1 once
each per episode, a few in two hundred: never per message, nothing for talking more or following people) are told
from it here."""

from megastructure.sim.constants import SAY_R, sec

ADDRESSED_S = sec(6)          # "answered someone": said something this soon after being spoken to
ASKED_S = sec(30)             # an ask stays open this long
SOCIAL = ("greeted_someone", "answered_someone", "gave_item", "traded", "helped", "worked_near_player")
NEAR_WORK = 6


class SocialMemory:
    def __init__(self, me: str, items):
        self.me, self.items = me, tuple(sorted(items, key=len, reverse=True))     # (longest first: "stone_block" before "stone")
        self.said_to_me = -10 ** 9
        self.asked, self.asked_by, self.asked_at = None, None, -10 ** 9
        self.got_from: set[str] = set()       # who gave me something
        self.gave_to: set[str] = set()        # who I gave something to
        self.got_now: dict = {}               # what was given to me since the last look (not "collected")

    def see(self, obs: dict):
        tick = obs["tick"]
        self.got_now = {}
        for e in obs.get("events", []):
            who = e.get("from")
            if e.get("type") in ("say", "shout") and who and who != self.me:
                self.said_to_me = e.get("tick", tick)
                text = str(e.get("text", "")).lower().replace(" ", "_")
                it = next((i for i in self.items if i in text), None)
                if it:
                    self.asked, self.asked_by, self.asked_at = it, who, e.get("tick", tick)
            elif e.get("type") == "give" and e.get("to") == self.me and who:
                self.got_from.add(who)
                for k, n in (e.get("items") or {}).items():
                    self.got_now[k] = self.got_now.get(k, 0) + n

    def ask_open(self, tick) -> str | None:
        return self.asked if self.asked and tick - self.asked_at <= ASKED_S else None

    def flags(self, obs: dict) -> tuple:
        """(spoken to just now, an ask open for something I carry, given something by someone this episode)"""
        tick, inv = obs["tick"], obs["self"].get("inventory", {})
        ask = self.ask_open(tick)
        return (float(tick - self.said_to_me <= ADDRESSED_S), float(bool(ask and inv.get(ask))), float(bool(self.got_from)))

    def gift(self, obs: dict) -> str | None:
        """What `give` hands over: what was asked for, if I carry it; else the thing I carry most of."""
        inv = obs["self"].get("inventory", {})
        ask = self.ask_open(obs["tick"])
        if ask and inv.get(ask):
            return ask
        return max(inv, key=inv.get) if inv else None

    def achieved(self, tool: str, args: dict, ok: bool, tick: int, near: float, to: str | None = None) -> list:
        """The social achievements an action that came back earned (the Ledger unlocks each once). near: how far the
        nearest other person is (tiles)."""
        if not ok:
            return []
        out = []
        if tool in ("say", "shout") and near <= SAY_R:
            out.append("answered_someone" if tick - self.said_to_me <= ADDRESSED_S else "greeted_someone")
        elif tool == "give" and to:
            out.append("gave_item")
            if args.get("item") == self.ask_open(tick) and to == self.asked_by:
                out.append("helped")
                self.asked = None
            if to in self.got_from:
                out.append("traded")
            self.gave_to.add(to)
        elif tool in ("dig", "craft", "place") and near <= NEAR_WORK:
            out.append("worked_near_player")
        return out

    def traded_back(self) -> bool:
        """Given something by someone I had given to (the other half of a trade)."""
        return bool(self.got_now) and bool(self.got_from & self.gave_to)
