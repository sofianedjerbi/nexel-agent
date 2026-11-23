"""What the player feels and sees of danger, said plainly so the model can choose (fight, run, eat, hide): beings near
with how far, which way, what they do to you and what they drop; your health trend and who hurt you; night; hunger;
your last death and its cause. Information only: nothing here picks an action.
"""

import time
from collections import deque

from megastructure.sim.entities import CREATURES
from megastructure.sim.specs import LEGEND
from megastructure.tiles import FIST_DAMAGE, FOOD, WEAPONS

CH = {v["char"]: k for k, v in CREATURES.items()}
WHAT = {k: LEGEND.get(v["char"], k).split(" (", 1)[1].rstrip(")") if " (" in LEGEND.get(v["char"], "") else ""
        for k, v in CREATURES.items()}


def compass(dx, dy) -> str:
    ns = "N" if dy < 0 else "S" if dy > 0 else ""
    ew = "W" if dx < 0 else "E" if dx > 0 else ""
    return (ns + ew) or "here"


class Body:
    """Health over the last half minute and what hurt it (from the observations' events)."""

    def __init__(self):
        self.hp: deque = deque(maxlen=40)          # (t, hp)
        self.hurts: deque = deque(maxlen=6)        # (t, who, damage, weapon)
        self.winds: dict = {}                      # creature id -> (t, dir) of a charge wound up toward you
        self.death: dict | None = None             # {t, cause, by, at}

    def feel(self, obs, now=None):
        now = now or time.time()
        me = obs["self"]
        self.hp.append((now, me["hp"]))
        for e in obs.get("events", []):
            if e.get("type") == "attack" and e.get("target") == me.get("name"):
                self.hurts.append((now, str(e.get("attacker")), e.get("damage"), e.get("weapon")))
            elif e.get("type") == "windup" and "id" in e:
                self.winds[e["id"]] = (now, e.get("dir"))
            elif e.get("type") == "death" and e.get("name") == me.get("name"):
                self.death = {"t": now, "cause": e.get("cause"), "by": e.get("by"), "at": (me["x"] + e.get("dx", 0), me["y"] + e.get("dy", 0))}
            elif e.get("type") == "fall" and e.get("name") == me.get("name") and e.get("damage"):
                self.hurts.append((now, "a fall", e.get("damage"), f"{e.get('layers')} layers"))

    def trend(self, now=None) -> int:
        now = now or time.time()
        old = next((h for t, h in self.hp if now - t <= 15), None)
        return (self.hp[-1][1] - old) if old is not None and self.hp else 0


def danger_lines(obs, body: Body | None = None, now=None) -> list[str]:
    now = now or time.time()
    me = obs["self"]
    inv = me["inventory"]
    out = []
    beings = []
    for e in obs["entities"][:6]:
        d = abs(e["dx"]) + abs(e["dy"])
        pos = f"({me['x'] + e['dx']},{me['y'] + e['dy']}) {d} tiles {compass(e['dx'], e['dy'])}"
        if e["kind"] == "agent":
            hit = any(who == e["name"] and now - t < 120 for t, who, _, _ in (body.hurts if body else ()))
            beings.append(f"{e['name']} #{e['id']} {pos}, hp {e['hp']}, holding {e.get('holding', '?')}, {e.get('doing', '')}"
                          + (" - hit you lately" if hit else ""))
            continue
        c = CREATURES.get(e["kind"], {})
        bits = [f"{e['kind']} #{e['id']} {pos}, hp {e['hp']}"]
        if c.get("damage"):
            bits.append(f"hits {c['damage']}")
        if WHAT.get(e["kind"]):
            bits.append(WHAT[e["kind"]])
        drops = [x[0] for x in c.get("drops", ()) if x[1] >= 0.5]
        if drops:
            bits.append("drops " + "+".join(drops))
        w = body.winds.get(e["id"]) if body else None
        if w and now - w[0] < 4:
            bits.append("WINDING UP A CHARGE" + (f" toward {compass(*w[1])}" if w[1] else ""))
        beings.append(", ".join(bits))
    if beings:
        out.append("Beings: " + "; ".join(beings))
    if body:
        fresh = [(t, who, dmg, wpn) for t, who, dmg, wpn in body.hurts if now - t < 20]
        tr = body.trend(now)
        if fresh or tr <= -3:
            who = sorted({w for _, w, _, _ in fresh})
            out.append(f"HURT: health {me['hp']}/20 ({tr:+d} in 15 s)" + (f", by {', '.join(who)}" if who else "") + ".")
        if body.death and now - body.death["t"] < 900:
            d = body.death
            out.append(f"You died {int((now - d['t']) // 60)} min ago at ({d['at'][0]},{d['at'][1]}): {d['cause']}"
                       + (f" ({d['by']})" if d.get("by") else "") + "; your pack was dropped there.")
    if "(night)" in str(obs.get("time", "")):
        out.append("Night: strawmen walk at you; firelight (campfire, torch) keeps them off; walls with a door, or a roof, shelter you.")
    food = [f"{k} +{FOOD[k]}" for k in inv if k in FOOD]
    if me["hunger"] <= 8:
        out.append(f"Hungry: food {me['hunger']}/20 (at 0 you lose health; at 9+ you heal). Food you carry: {', '.join(food) or 'none'}.")
    elif me["hp"] <= 12 and me["hunger"] < 9 and food:
        out.append(f"You heal only with food 9+: you carry {', '.join(food)}.")
    return out


def rules_text() -> str:
    """For the stable prefix: what kills, what heals, how fights go."""
    weapons = ", ".join(f"{k} {v}" for k, v in WEAPONS.items())
    return (f"Survival: at health 0 you die and drop your pack. You heal while food is 9+; food 0 hurts; eat ITEM. "
            f"Fight: attack ID [with WEAPON], within 3 tiles (fists {FIST_DAMAGE}; {weapons}). Beings only reach "
            "you up close; distance and walls keep them off.")
