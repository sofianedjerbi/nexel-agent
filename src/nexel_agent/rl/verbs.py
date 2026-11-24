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
    if tool == "ring":
        return [("place", {"x": x, "y": y, "item": it}) for x, y, it in args["pieces"]] or [("noop", {})]
    if tool == "attack" and args.get("fight"):
        return [cmd] * FIGHT
    return [cmd]                         # (attack at someone who has not hit it: a single swing)
