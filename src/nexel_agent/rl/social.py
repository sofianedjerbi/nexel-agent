"""Who spoke to the player, what they asked for, who gave it what: read from the observation's events, the same in
the sim and on the live server. The policy sees three flags of it (rl/encode.py); the social achievements (+1 once
each per episode, a few in two hundred: never per message, nothing for talking more or following people) are told
from it here."""

from nexel_worlds.sim.constants import SAY_R, sec
from .facts import Attack, Death, Drop, Give, Say, Took, events, handed, named

CELL_ITEMS = ("plank", "roof", "stone_block", "dirt")     # (what fills a cell; anything else stands in one: rl/plan.py BLOCKS)


def laid_as(item) -> str:
    """How a piece laid up a storey is kept in mind: as the cell's block or as a thing standing in it. Kept by its
    place alone, a plank laid read as the wall on it laid too: the tower's wall never went up, its flight ran short."""
    return "block" if item in CELL_ITEMS else "thing"

ADDRESSED_S = sec(6)          # "answered someone": said something this soon after being spoken to
TALK_GAP = sec(30)            # a chat cooldown, the same for every speaker: a line once spoken to, or 30 s after the last
HIT_S = sec(15)               # "hit me lately", "I hit them lately"
ASKED_S = sec(30)             # an ask stays open this long
SOCIAL = ("greeted_someone", "answered_someone", "gave_item", "traded", "helped", "worked_near_player")
NEAR_WORK = 6


class SocialMemory:
    def __init__(self, me: str, items):
        self.me, self.items = me, tuple(items)
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
        self.home = None                      # (x, y) of the home's inside, once chosen (rl/build.py)
        self.up_laid = set()                  # (x, y, z) laid up a storey, (x, y, "stairs"): the view shows the land only
        self.began: dict = {}                 # name: who struck the first blow of the incident with them
        self.drops: dict = {}                 # name: [(tick, items)] they let fall lately (facts.handed)
        self.where: dict = {}                 # each person: where I last saw them, (x, y, hp)
        self.words_tell = False               # live with a words model: it tells which lines are asks (rl/acts.py), not a name in them

    def see(self, obs: dict):
        tick, me = obs["tick"], obs.get("self") or {}
        self.got_now, self.got_by = {}, {}
        for b in obs.get("entities", []):
            if b.get("kind") == "agent" and b.get("name"):
                self.where[b["name"]] = (me.get("x", 0) + b["dx"], me.get("y", 0) + b["dy"], b.get("hp", 0))
        for ev in events(obs):
            t = tick if ev.tick is None else ev.tick
            match ev:
                case Say(src=who) if who == self.me:
                    self.my_last = t
                case Say(src=who) if who:
                    self.said_to_me = t
                    if not self.words_tell and (it := next(iter(named(ev.text, self.items)), None)):
                        self.asked, self.asked_by, self.asked_at = it, who, t     # (the sim's neighbours: their lines are asks)
                case Attack(by=by, target=tgt) if tgt == self.me and by:
                    if not self.lately(by, t):                       # (a new incident: they began it)
                        self.began[by] = by
                    self.hit_me[by] = t
                    self.last_hit = (by, ev.id, t)                  # (a creature's blow says its id)
                case Attack(by=by, target=tgt) if by == self.me and tgt:
                    if not self.lately(tgt, t):                      # (a new incident: it began it)
                        self.began[tgt] = self.me
                    self.i_hit[tgt] = t
                case Death(name=who) if who and who != self.me:      # (a hitter who fell is dealt with: no blow back
                    self.hit_me.pop(who, None)                       # at them when they come back - live trial 8: Rook
                    if self.last_hit[0] == who:                      # killed Mira 5 times, each time she respawned)
                        self.last_hit = (None, None, -10 ** 9)
                case Drop(name=who) if who and who != self.me:
                    self.drops.setdefault(who, []).append((t, dict(ev.items)))
                case Took(name=who) if who == self.me:               # (dropped by them, picked up: a gift, as a give)
                    for giver, k, n in handed(self.drops, ev.items, t, sec(30)):
                        self.got_from.add(giver)
                        self.got_now[k] = self.got_now.get(k, 0) + n
                        self.got_by.setdefault(giver, {})[k] = self.got_by.get(giver, {}).get(k, 0) + n
                case Give(src=who, to=to) if to == self.me and who:
                    self.got_from.add(who)
                    for k, n in ev.items.items():
                        self.got_now[k] = self.got_now.get(k, 0) + n
                        self.got_by.setdefault(who, {})[k] = self.got_by.get(who, {}).get(k, 0) + n

    def heard_ask(self, who, item, tick):
        """The words model told this line an ask (rl/acts.py): it opens, as a neighbour's does in the sim."""
        self.asked, self.asked_by, self.asked_at = item, who, tick

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

    def lately(self, name, tick) -> bool:
        """A blow between it and `name`, either way, HIT_S ago at most: an incident still on."""
        return tick - max(self.hit_me.get(name, -10 ** 9), self.i_hit.get(name, -10 ** 9)) <= HIT_S

    def they_began(self, name, tick) -> bool:
        """`name` struck first in the incident still on (a fight back at them is defence, not a feud it began)."""
        return self.lately(name, tick) and self.began.get(name) == name

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
        if tool == "dig" and ok and "z" in args and "x" in args:          # (a cell dug, by the build verb: rl/build.py)
            self.up_laid.add((args["x"], args["y"], args["z"], "dug"))
        if tool == "place" and ok and "x" in args:          # (what it laid where the view does not show: rl/build.py)
            if args.get("item") == "stairs":
                self.up_laid.add((args["x"], args["y"], "stairs"))
            elif "z" in args:                   # (a block in the cell or a thing standing there: a plank and the
                self.up_laid.add((args["x"], args["y"], args["z"], laid_as(args.get("item"))))   # wall on it, both)
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
