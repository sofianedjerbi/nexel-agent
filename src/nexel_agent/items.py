"""A queued command (Item) and two helpers on the way in: the pack after a reply, and a being named in a command."""

from dataclasses import dataclass


def inv_after(obs, reply) -> dict:
    """The pack after an action: the observation's, plus the reply's changes."""
    inv = dict(obs["self"]["inventory"])
    for k, n in ((reply.get("result") or {}).get("changed") or {}).get("inventory", {}).items():
        inv[k] = inv.get(k, 0) + n
    return inv


def by_id(raw, obs):
    """'attack Mira' / 'give tusker ...': a name or kind in view stands for its id (the nearest one)."""
    if not isinstance(raw, str) or not obs:
        return raw
    w = raw.split()
    if len(w) >= 2 and w[0].lower() in ("attack", "give", "feed", "milk", "name") and not w[1].lstrip("#-").isdigit():
        who = w[1].lstrip("#").lower()
        e = next((e for e in obs.get("entities", []) if who in (str(e.get("name", "")).lower(), e["kind"])), None)
        if e:
            return " ".join([w[0], str(e["id"])] + w[2:])
    return raw


@dataclass
class Item:
    tool: str
    args: dict
    source: str                 # model | chat
    why: str = ""
    at: tuple | None = None     # where the plan expected to stand for a relative (dir) command
    epoch: int = 0
    plan: int = 0               # which answer it came from (repeats, failures of the same plan)
