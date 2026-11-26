"""How one action becomes the game's tool calls (the env and the live body alike)."""

FIGHT = 12                          # attack: swings at one target, till it falls or gets away (~5 s at most)


def split(cmd) -> list:
    """The tool calls of one action: itself; or, climb, the game's own way up a step: the block over your head dug
    (if any), the land in front at head height (if any), then a jump that way; or, ring, one place after another
    (rl/ring.py); or, attack, a fight: swings at that target one after the other (a swing that misses ends it) - at
    whoever hit it lately; at anyone else, one swing."""
    tool, args = cmd
    if tool == "climb":
        w = {"with": args["with"]} if args.get("with") else {}        # (the tool for the land: rl/actions.py tool_for)
        return [("dig", {"dir": "up", **w}), ("dig", {"dir": args["dir"], "z": args["z"], **w}), ("jump", {"dir": args["dir"]})]
    if tool == "ring":                   # (build's: a walk to where it lays them first, `go`)
        go = [("move_to", args["go"])] if args.get("go") else []
        return go + [("dig", {"dir": p[4], "z": p[3], "x": p[0], "y": p[1], **({"with": p[5]} if p[5] else {})})
                     if p[2] == "dig" else ("dig", {"x": p[0], "y": p[1], "with": p[3]}) if p[2] == "till" else ("place", {"x": p[0], "y": p[1], "item": p[2], **({"z": p[3]} if len(p) > 3 else {})})
                     for p in args["pieces"]] or [("noop", {})]
    if tool == "attack" and args.get("fight"):
        return [cmd] * FIGHT
    return [cmd]                         # (attack at someone who has not hit it: a single swing)
