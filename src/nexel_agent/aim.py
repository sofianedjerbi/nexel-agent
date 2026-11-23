"""The body's plumbing around results: an action that just failed is not resent at once, and walks that find no path
are counted (escape.stuck tells the model it is stuck). No action is chosen here."""

BACKOFF_S = 1.0                                  # an action that just failed is not sent again this soon


def just_failed(memory, tool, args, now) -> bool:
    last = memory.get("last_fail")
    return bool(last) and last[:2] == (tool, args) and now - last[2] < BACKOFF_S


def note_outcome(memory, obs, item, reply, t1):
    """The last failure (backoff) and no_path in a row."""
    ok, err = bool(reply.get("ok")), reply.get("error")
    if not ok and err not in ("interrupted", "rate_limited", "hands_full"):
        memory["last_fail"] = (item.tool, item.args, t1)
    if item.tool in ("move_to", "step", "walk", "jump") or err == "no_path":
        memory["no_path"] = memory.get("no_path", 0) + 1 if err == "no_path" else 0
