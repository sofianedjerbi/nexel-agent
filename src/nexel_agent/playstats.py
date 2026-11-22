"""What a real-time player run measured: seconds between actions, idle share, model latency, prompt size."""

from collections import Counter


def summary(busy, calls, wall, stats) -> dict:
    """Seconds between action starts, the share of time with nothing running on the server, model latency."""
    busy = sorted(busy)
    covered, end = 0.0, None
    for s, e, _ in busy:
        if end is None or s > end:
            covered += e - s
            end = e
        elif e > end:
            covered += e - end
            end = e
    starts = [s for s, _, _ in busy]
    gaps = sorted(b - a for a, b in zip(starts, starts[1:]))
    lat = sorted(c["latency_s"] for c in calls if c.get("kind") == "call")
    chat = sorted(c["latency_s"] for c in calls if c.get("kind") == "chat")
    first = sorted(c["first_cmd_s"] for c in calls if c.get("kind") == "call" and c.get("first_cmd_s"))
    pct = lambda xs, q: round(xs[min(len(xs) - 1, int(len(xs) * q))], 2) if xs else None  # noqa: E731
    span = (busy[-1][1] - busy[0][0]) if busy else wall
    toks = [c["usage"].get("prompt_tokens") for c in calls if c.get("usage")]
    cached = [c["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for c in calls if c.get("usage")]
    return {"actions": len(busy), "by_source": dict(Counter(src for _, _, src in busy)),
            "s_between_actions_p50": pct(gaps, 0.5), "s_between_actions_p95": pct(gaps, 0.95),
            "idle_pct": round(100 * (1 - covered / span), 1) if span > 0 else None,
            "model_calls": len(lat), "model_latency_p50": pct(lat, 0.5), "model_latency_p95": pct(lat, 0.95),
            "first_action_after_s_p50": pct(first, 0.5), "chat_latency_p50": pct(chat, 0.5),
            "prompt_tokens_p50": pct(sorted(toks), 0.5), "cached_tokens_p50": pct(sorted(cached), 0.5),
            "stats": dict(stats)}
