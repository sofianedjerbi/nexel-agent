"""Who spoke to the player, what they asked for, who gave it what: read from the observation's events, the same in
the sim and on the live server. The policy sees three flags of it (rl/encode.py); the social achievements (+1 once
each per episode, a few in two hundred: never per message, nothing for talking more or following people) are told
from it here."""

from megastructure.sim.constants import SAY_R, sec

ADDRESSED_S = sec(6)          # "answered someone": said something this soon after being spoken to
TALK_GAP = sec(30)            # a chat cooldown, the same for every speaker: a line once spoken to, or 30 s after the last
HIT_S = sec(15)               # "hit me lately", "I hit them lately"
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
        self.got_by: dict = {}                # the same, by giver
        self.hit_me: dict = {}                # who hit me, when
        self.i_hit: dict = {}                 # whom I hit, when
        self.my_last = -10 ** 9               # when I last said something
        self.last_hit = (None, None, -10 ** 9)   # who hit me last: (name, id, tick)

    def see(self, obs: dict):
        tick = obs["tick"]
        self.got_now, self.got_by = {}, {}
        for e in obs.get("events", []):
            who = e.get("from")
            if e.get("type") in ("say", "shout") and who and who != self.me:
                self.said_to_me = e.get("tick", tick)
                text = str(e.get("text", "")).lower().replace(" ", "_")
                it = next((i for i in self.items if i in text), None)
                if it:
                    self.asked, self.asked_by, self.asked_at = it, who, e.get("tick", tick)
            elif e.get("type") in ("say", "shout") and who == self.me:
                self.my_last = e.get("tick", tick)
            elif e.get("type") == "attack":
                if e.get("target") == self.me and e.get("attacker"):
                    self.hit_me[e["attacker"]] = e.get("tick", tick)
                    self.last_hit = (e["attacker"], e.get("id"), e.get("tick", tick))      # (a creature's blow says its id)
                elif e.get("attacker") == self.me and e.get("target"):
                    self.i_hit[e["target"]] = e.get("tick", tick)
            elif e.get("type") == "give" and e.get("to") == self.me and who:
                self.got_from.add(who)
                for k, n in (e.get("items") or {}).items():
                    self.got_now[k] = self.got_now.get(k, 0) + n
                    self.got_by.setdefault(who, {})[k] = self.got_by.get(who, {}).get(k, 0) + n

    def can_talk(self, tick) -> bool:
        """The chat cooldown (an interface limit, the same for anyone who talks): spoken to since my last line, or my
        last line TALK_GAP ago."""
        return self.said_to_me > self.my_last or tick - self.my_last >= TALK_GAP

    def people_flags(self, obs: dict, n: int) -> list:
        """For the n nearest beings: (hit me lately, I hit them lately); people only (a creature's kind says enough)."""
        tick, out = obs["tick"], []
        for e in obs.get("entities", [])[:n]:
            name = e.get("name") if e.get("kind") == "agent" else None
            out += [float(bool(name) and tick - self.hit_me.get(name, -10 ** 9) <= HIT_S),
                    float(bool(name) and tick - self.i_hit.get(name, -10 ** 9) <= HIT_S)]
        return out + [0.0] * (2 * n - len(out))

    def hit_me_lately(self, name, tick) -> bool:
        return bool(name) and tick - self.hit_me.get(name, -10 ** 9) <= HIT_S

    def hitter(self, obs: dict):
        """The being that hit me last (HIT_S ago at most), if it is in view."""
        name, cid, t = self.last_hit
        if obs["tick"] - t > HIT_S:
            return None
        return next((e for e in obs.get("entities", []) if (cid is not None and e.get("id") == cid)
                     or (e.get("kind") == "agent" and e.get("name") == name)), None)

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
