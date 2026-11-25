"""The build judge (#26, v1: does it work?): the game's own reading of a build (nexel_worlds.sim.builds.report), as
checks a player would name, each passed or not, with the fact behind it. No weights: the grade is the checks passed.
v2 (does it make sense, #31) and v3 (does it look good, #32) add checks here, on the same report."""

from nexel_worlds.sim import builds

ROOFED = 0.9                # of the inside under a roof
AREA = 4                    # tiles inside, at least: room to stand, sleep and turn round
CHECKS = ("closed", "door", "roofed", "lit", "bed", "room_enough")


def verdict(world, x, y) -> dict:
    """The build around (x, y): {"grade": checks passed, "of": how many, "checks": {name: bool}, "failed": [names],
    "report": the game's facts}."""
    r = builds.report(world, x, y)
    c = r["closed"]
    checks = {"closed": c,
              "door": c and r["doors"] > 0,
              "roofed": c and r["roofed"] >= ROOFED,
              "lit": c and r["lights"] > 0,
              "bed": c and r["contents"].get("bed", 0) > 0,
              "room_enough": c and r["area"] >= AREA}
    return {"grade": sum(checks.values()), "of": len(CHECKS), "checks": checks,
            "failed": [k for k in CHECKS if not checks[k]], "report": r}


def told(v: dict) -> str:
    """The verdict in words, for the mind's facts: "closed, a door, no roof (40%), no light, no bed (grade 2 of 6)"."""
    r, ch = v["report"], v["checks"]
    if not r["closed"]:
        return "not closed"
    parts = ["closed", "a door" if ch["door"] else "no door",
             "roofed" if ch["roofed"] else f"no roof ({r['roofed']:.0%} covered)",
             "lit" if ch["lit"] else "no light", "a bed" if ch["bed"] else "no bed",
             f"{r['area']} tiles inside"]
    return ", ".join(parts) + f" (grade {v['grade']} of {v['of']})"
