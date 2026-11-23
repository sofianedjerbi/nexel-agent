"""Facts read from the game's observation, in one place: one parser per event type (each returns a small record), one
wording of an event, one distance, one "is this a creature", one "which items does this line name". Every reader in
rl/ goes through here: the social memory, what is happening now (voice.py Now: the words and the planner), the brain
feed, the live body, the sim's neighbours.

Distance. The game's reach is a circle round you, as the crow flies: megastructure sim/bodies.py near() is
dx² + dy² (+ the height, a layer half a tile) <= (r + .5)². So dist() is Euclidean and within() is the server's own
test, on the map (an observation gives a being's dx, dy, not its height). What a player sees and hears is a square of
VIEW_R / SAY_R round it (sim/core.py emit, sim/observe.py): the server has applied it already to what an observation
holds, nothing here measures it again."""

import math
import re
from typing import NamedTuple


HOSTILE = ("tusker", "pincer", "strawman", "wyrm", "mossback", "sporeling")      # the creatures that come at you


# ------------------------------------------------------------------ where

def dist(b: dict) -> float:
    """How far a being of the observation is, in tiles, as the server measures reach."""
    return math.hypot(b["dx"], b["dy"])


def within(b: dict, r: float, rise: float = 0.0) -> bool:
    """The server's reach test (sim/bodies.py near): r tiles and the half tile to the far tile's edge; `rise`, the
    height between you in tiles, is distance too."""
    return b["dx"] ** 2 + b["dy"] ** 2 + rise ** 2 <= (r + .5) ** 2


def compass(dx, dy) -> str:
    d = round(math.hypot(dx, dy))
    if d <= 1:
        return "right here"
    return f"{d} tiles {('E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE')[round(math.atan2(dy, dx) / (math.pi / 4)) % 8]}"


def people(obs: dict) -> list:
    """The other players in view, nearest first."""
    return sorted((b for b in obs.get("entities", []) if b.get("kind") == "agent"), key=dist)


# ------------------------------------------------------------------ who, what

def is_creature(name) -> bool:
    """An event names a player by name, a creature as "a tusker" (its blow) or "tusker#12" (a blow on it)."""
    return str(name).startswith("a ") or "#" in str(name)


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


def _at(e):
    return (e["x"], e["y"]) if "x" in e and "y" in e else None


PARSERS = {
    "say": _say, "shout": _say,
    "attack": lambda e: Attack(e.get("attacker"), e.get("target"), e.get("damage") or 0, e.get("weapon"), e.get("id"), e.get("tick")),
    "give": lambda e: Give(e.get("from"), e.get("to"), dict(e.get("items") or {}), e.get("tick")),
    "death": lambda e: Death(e.get("name"), e.get("cause"), e.get("tick"), _at(e), e.get("dropped")),
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
            return f"{who(ev.name)} died" + (f" ({ev.cause})" if ev.cause else "")
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
