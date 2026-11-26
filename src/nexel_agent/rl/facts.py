"""Facts read from the game's observation, in one place: one parser per event type (each returns a small record), one
wording of an event, one distance, one "is this a creature", one "which items does this line name". Every reader in
rl/ goes through here: the social memory, what is happening now (voice.py Now: the words and the planner), the brain
feed, the live body, the sim's neighbours.

Distance. The game's reach is a circle round you, as the crow flies: nexel_worlds sim/bodies.py near() is
dx² + dy² (+ the height, a layer half a tile) <= (r + .5)². So dist() is Euclidean; whether a thing is in reach is
asked of the game itself (sim/bodies.py reach_of, in_reach: rl/actions.py), never worked out again here. What a thing takes is read from the game's
own recipe table (needs), never a list of ours. What a player sees and hears is a square of
VIEW_R / SAY_R round it (sim/core.py emit, sim/observe.py): the server has applied it already to what an observation
holds, nothing here measures it again."""

import math
import re
from typing import NamedTuple

from nexel_worlds.sim.entities import CREATURES
from nexel_worlds.tiles import RECIPES


HOSTILE = tuple(k for k, c in CREATURES.items() if c.get("damage"))      # the creatures that come at you (the game's: they do damage)


# ------------------------------------------------------------------ where

def dist(b: dict) -> float:
    """How far a being of the observation is, in tiles, as the server measures reach."""
    return math.hypot(b["dx"], b["dy"])


def compass(dx, dy) -> str:
    d = round(math.hypot(dx, dy))
    if d <= 1:
        return "right here"
    return f"{d} tiles {('E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE')[round(math.atan2(dy, dx) / (math.pi / 4)) % 8]}"


def people(obs: dict) -> list:
    """The other players in view, nearest first."""
    return sorted((b for b in obs.get("entities", []) if b.get("kind") == "agent"), key=dist)


# ------------------------------------------------------------------ who, what

def kind_of(ref, name=None):
    """What an event names, by its kind: "agent" for a person, else the creature's ("tusker"), from who it says did it
    ({kind, id, name}, nexel-worlds#119). An event that names it only in words (the game before it: the agent's pin
    31963d92) is read from them: a creature is "a tusker" (its blow) or "tusker#12" (a blow on it), a person by name."""
    if ref:
        return ref.get("kind")
    if not name:
        return None
    name = str(name)
    return name[2:] if name.startswith("a ") else name.split("#")[0] if "#" in name else "agent"


def said(ref, blow: bool):
    """Who an event names, in the words the game's events used: a person by name, a creature as "a tusker" when it
    strikes, "tusker#12" when it is struck (the names every reader here keys on: rl/social.py, rl/folk.py)."""
    if not ref:
        return None
    if ref.get("kind") == "agent":
        return ref.get("name")
    return f"a {ref.get('kind')}" if blow else f"{ref.get('kind')}#{ref.get('id')}"


def plain(name) -> str:
    """A name as one says it: "tusker#12" is "a tusker"."""
    name = str(name or "someone")
    return "a " + name.split("#")[0] if "#" in name else name


def named(text: str, items) -> list:
    """The items a line names, the longest name first ("stone block" before "stone"); whole words only ("more" is
    not "ore"), a plural s allowed."""
    low = str(text).lower().replace("_", " ")
    words = set(re.findall(r"[a-z]+", low))
    return [k for k in sorted(items, key=len, reverse=True) if (p := k.replace("_", " ")) in low
            and (" " in p or k in words or k + "s" in words)]


def counted(d: dict) -> str:
    return ", ".join(f"{n:g} {k.replace('_', ' ')}" for k, n in d.items())


def pack(obs: dict) -> str:
    """Everything it carries, most first (the observation's own inventory: never cut, a single wood counts)."""
    inv = (obs.get("self") or {}).get("inventory") or {}
    return counted(dict(sorted(inv.items(), key=lambda kv: -kv[1]))) or "nothing"


def ground(obs: dict) -> list:
    """The piles lying in view - drops, loot, a fallen player's pack (sim/observe.py "items") - nearest first:
    (dx, dy, {item: n})."""
    piles = [(p["dx"], p["dy"], dict(p["items"])) for p in obs.get("items") or [] if p.get("items")]
    return sorted(piles, key=lambda p: math.hypot(p[0], p[1]))


# ------------------------------------------------------------------ what a thing takes

def _recipe(r) -> str:
    return counted(r.inputs) + (f" at a {r.station}" if r.station else "")


def short(r, have: dict) -> dict:
    """What of recipe `r`'s inputs the pack `have` lacks, in the recipe's order: {item: n}. (The game's can_craft
    answers the same in newer nexel-worlds; the pinned one has no can_craft yet.)"""
    return {k: n - have.get(k, 0) for k, n in r.inputs.items() if have.get(k, 0) < n}


def needs(item: str, obs: dict, known=()) -> str:
    """What owning `item` takes, from the game's own recipe table (tiles.py RECIPES): how many it has, the
    recipe and what of it is short - one step down for a short input that is made too -, or that no recipe makes it
    (and, `known` given, that no item goes by that name)."""
    me = obs.get("self") or {}
    have, r = me.get("inventory") or {}, RECIPES.get(item)
    out = f"you have {have.get(item, 0)}"
    if r is None:
        return out + ", no recipe makes it" + ("" if not known or item in known else ", and no item you know goes by that name")
    out += "; made from " + _recipe(r)
    if miss := short(r, have):
        down = [f"{k} is made from {_recipe(RECIPES[k])}" for k in miss if k in RECIPES]
        return out + "; you lack " + counted(miss) + "".join(f" ({d})" for d in down)
    far = r.station and r.station not in (me.get("stations") or ())
    return out + (f"; no {r.station} in reach" if far else "; you can make it now")


# ------------------------------------------------------------------ events

class Say(NamedTuple):
    src: str | None
    text: str
    shout: bool = False
    tick: int | None = None


class Attack(NamedTuple):
    by: str | None
    target: str | None
    damage: float = 0
    weapon: str | None = None
    id: int | None = None              # a creature's blow says which creature
    tick: int | None = None
    by_kind: str | None = None         # "agent" for a person, else the creature's kind (kind_of)
    target_kind: str | None = None

    @property
    def by_person(self) -> bool:
        return self.by_kind == "agent"

    @property
    def on_person(self) -> bool:
        return self.target_kind == "agent"


class Give(NamedTuple):
    src: str | None
    to: str | None
    items: dict
    tick: int | None = None


class Death(NamedTuple):
    name: str | None
    cause: str | None = None           # the game's own words ("killed by Bob", "starved"): never added to
    tick: int | None = None
    at: tuple | None = None            # (the sim's own events only: where, and the pack that fell there)
    dropped: dict | None = None
    code: str | None = None            # why, as the game files it: killed, bitten, burnt, fell, starvation, fire, admin, dev
    by: str | None = None              # who did it, as a blow names it ("Bob", "a tusker"): from the event's who


class Took(NamedTuple):                # (pick_up, the sim's own events: what was taken from the ground, where)
    name: str | None
    items: dict
    tick: int | None = None
    at: tuple | None = None


class Fell(NamedTuple):                # (creature_died)
    kind: str | None
    by: str | None = None
    id: int | None = None
    tick: int | None = None


class Craft(NamedTuple):
    name: str | None
    made: dict
    tick: int | None = None


class Join(NamedTuple):
    name: str | None
    tick: int | None = None


class Boss(NamedTuple):
    what: str | None
    tick: int | None = None


def _say(e):
    return Say(e.get("from"), str(e.get("text", "")), e.get("type") == "shout", e.get("tick"))


def _attack(e):
    """A blow: who struck whom by who/whom (nexel-worlds#119), else by the attacker/target words of an older game."""
    who, whom = e.get("who"), e.get("whom")
    by = said(who, True) if who else e.get("attacker")
    target = said(whom, False) if whom else e.get("target")
    return Attack(by, target, e.get("damage") or 0, e.get("weapon"), e.get("id"), e.get("tick"), kind_of(who, by), kind_of(whom, target))


def _at(e):
    return (e["x"], e["y"]) if "x" in e and "y" in e else None


PARSERS = {
    "say": _say, "shout": _say,
    "attack": lambda e: _attack(e),
    "give": lambda e: Give(e.get("from"), e.get("to"), dict(e.get("items") or {}), e.get("tick")),
    "death": lambda e: Death(e.get("name"), e.get("cause"), e.get("tick"), _at(e), e.get("dropped"), e.get("code"), said(e.get("who"), True)),
    "pick_up": lambda e: Took(e.get("name"), dict(e.get("items") or {}), e.get("tick"), _at(e)),
    "creature_died": lambda e: Fell(e.get("kind"), e.get("by"), e.get("id"), e.get("tick")),
    "craft": lambda e: Craft(e.get("name"), dict(e.get("made") or {}), e.get("tick")),
    "join": lambda e: Join(e.get("name"), e.get("tick")),
    "boss": lambda e: Boss(e.get("what"), e.get("tick")),
}


def read(e: dict):
    """One game event as its record, or None (a type nothing here reads)."""
    p = PARSERS.get(e.get("type"))
    return p(e) if p else None


def events(obs_or_list) -> list:
    """The records of an observation's events (or of a list of events), in order."""
    raw = obs_or_list.get("events", []) if isinstance(obs_or_list, dict) else obs_or_list
    return [ev for e in raw if (ev := read(e)) is not None]


# A death in words, from its code (nexel-worlds#171 sends no words): as the game's death screen says it (web/js/death.js WHY_DIED)
WHY = {"killed": "killed by {by}", "bitten": "bitten by {by}", "burnt": "burnt by a hob", "fell": "fell", "starvation": "starved",
       "fire": "burnt in a fire", "admin": "struck down", "dev": "struck down"}


def told(ev, who=plain) -> str:
    """An event in words. who: how a name is said (voice.py says "you" for the player itself)."""
    match ev:
        case Say():
            return f'{who(ev.src)}: "{ev.text[:80]}"'
        case Attack():
            return f"{who(ev.by)} hit {who(ev.target)}"
        case Give():
            return f"{who(ev.src)} gave {who(ev.to)} {counted(ev.items)}"
        case Death():
            why = WHY[ev.code].format(by=who(ev.by) if ev.by else "someone") if ev.code in WHY else ev.cause    # (an older game's: its words)
            return f"{who(ev.name)} died" + (f" ({why})" if why else "")
        case Fell():
            return f"a {ev.kind} fell" + (f" to {who(ev.by)}" if ev.by else "")
        case Craft():
            return f"{who(ev.name)} made {counted(ev.made)}"
        case Join():
            return f"{who(ev.name)} came in"
        case Boss():
            return f"the boss: {ev.what}"
        case Took():
            return f"{who(ev.name)} picked up {counted(ev.items)}"
    raise TypeError(ev)
