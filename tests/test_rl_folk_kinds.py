"""Who comes by (rl/folk.py, #12): a mixture of the kinds of person the live save's real players were, measured per
active minute (the issue's table): quiet ones a share (QUIET), rough ones a few (ROUGH), the rest social as before.
The sim's neighbours, counted the same way, land in each kind's range."""

from collections import Counter, defaultdict

from nexel_agent.rl import folk
from nexel_agent.rl.core import SimEnv

STEPS = 1500                                  # (about 17 minutes of the game's time: 4 neighbours coming and going)


def rates(seed=3):
    """Per kind, per minute it was there: lines said, asks, gifts, blows on the player, on creatures; and the share
    of that time it was within 8 tiles of the player."""
    env = SimEnv(ep_ticks=10 ** 7, roam=0, rivals=0, folk=4, shadow_pct=0)
    env.reset(seed)
    sim, me = env.sim, env.a
    kind_of = {f.a.name: f for f in env.folk}
    did, ticks, near = defaultdict(Counter), Counter(), Counter()
    submit = sim.submit

    def counted(a, tool, args):
        f = kind_of.get(a.name)
        if f is not None and f.on:
            k = did[f.kind]
            if tool == "say":
                k["says"] += 1
                k["asks"] += "spare" in args.get("text", "")
            elif tool == "give":
                k["gives"] += 1
            elif tool == "attack":
                k["hits_people" if args.get("target") == me.id else "hits_creatures"] += 1
        return submit(a, tool, args)
    sim.submit = counted
    for _ in range(STEPS):
        t = sim.tick
        env.step(0)
        me.hp, me.hunger = 20, 20                             # (the player stands and lives: who comes by is what is measured)
        for f in env.folk:
            if f.on:
                ticks[f.kind] += sim.tick - t
                near[f.kind] += (sim.tick - t) * (f.dist() <= 8)
    mins = {k: n / 600 for k, n in ticks.items()}
    return {k: {**{c: did[k][c] / mins[k] for c in ("says", "asks", "gives", "hits_people", "hits_creatures")},
                "near": near[k] / ticks[k], "minutes": mins[k]} for k in ticks}


def test_quiet_neighbours_are_as_the_live_save_s_players_and_the_social_stay_social():
    r = rates()
    q, s = r["quiet"], r["social"]
    assert q["minutes"] > 3 and s["minutes"] > 3, r
    # quiet (measured, humans' medians: says 0.11, gives 0, asks 0, hits on people 0, near 0; creatures hunted ~1)
    assert q["says"] < 0.5 and q["asks"] == 0 and q["gives"] == 0 and q["hits_people"] == 0 and q["near"] < 0.2, q
    # social: as before, near the player and talking, asking, giving
    assert s["near"] > 0.6 and s["says"] > q["says"] and s["asks"] > 0, s
    assert abs(folk.QUIET - 0.4) < 1e-9 and folk.ROUGH < 0.1


def test_a_rough_neighbour_hits_people(monkeypatch):
    monkeypatch.setattr(folk, "ROUGH", 1.0)
    r = rates()["rough"]
    assert r["hits_people"] > 0.1, r                            # (the measured one: half a blow a minute)
