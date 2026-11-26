"""Speech acts (#20): what a line heard does - asks for an item, offers one, asks how a thing is made, greets, says
bye, asks what it carries, or just talks - told by the words model itself in the same call that writes its reply,
never by a list of keywords (a line that names an item is not an ask: "a sword is 2 ingot" is an answer, #9). And what
its own line does: a promise to hand something over binds its body (rl/mind.py commit), a line that reads out its
pack goes unsaid unless someone asked what it carries, and a line that says it gave or made a thing goes unsaid when
the game shows no such deed (rl/now.py backs)."""

import json
import re

# (the words' reply, as the model is told to write it: rl/voice.py)
REPLY = ('Reply with JSON only: {"heard": [one label for each line said to you, in order: "ask ITEM" (they want an item '
         'from you), "offer ITEM" (they offer you one), "how ITEM" (they ask what it takes or where it is found), "pack" '
         '(they ask what you carry), "tell" (they tell you a fact), "greet", "bye", "chat"], "say": "your chat line, or empty to say nothing", '
         '"promise": "ITEM N" when your line says you will hand them something (and you carry it), else empty, '
         '"deed": "gave ITEM" or "made ITEM" when your line says you already gave them or made it, else empty}')
KINDS = ("ask", "offer", "how", "pack", "tell", "greet", "bye", "chat")


def item_of(word, items) -> str | None:
    w = str(word or "").strip().lower().replace(" ", "_")
    return w if w in items else w[:-1] if w.endswith("s") and w[:-1] in items else None


def _json(out: str) -> tuple:
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        d = None
    return out, d if isinstance(d, dict) else None


def deed(out: str, items) -> tuple | None:
    """("gave" | "made", item) its line claims, from the same reply, or None."""
    kind, _, rest = str((_json(out)[1] or {}).get("deed") or "").strip().lower().partition(" ")
    return (kind, item_of(rest, items)) if kind in ("gave", "made") and item_of(rest, items) else None


def read(out: str, n_heard: int, items) -> tuple:
    """The model's reply -> (line, [(kind, item)] one per line heard, (item, n) promised or None). A reply that is no
    JSON is taken as the line itself, with nothing heard told apart (as before the acts)."""
    out, d = _json(out)
    if d is None:
        return out, [], None
    heard = []
    for lab in (d.get("heard") or [])[:n_heard]:
        kind, _, rest = str(lab).strip().lower().partition(" ")
        if kind in KINDS:
            heard.append((kind, item_of(rest, items) if kind in ("ask", "offer", "how") else None))
    promise = None
    p = str(d.get("promise") or "").split()
    if p and (it := item_of(" ".join(w for w in p if not w.isdigit()), items)):
        promise = (it, next((int(w) for w in p if w.isdigit()), 1))
    return str(d.get("say") or ""), heard, promise


def pack_dump(line: str, items, pack: dict | None = None) -> bool:
    """Does the line read out its pack: two or more items it names with a count ("stone_block 26, dirt 13")? With
    the pack, only counts it carries are its pack: "6 wood and 3 fiber" to "how is a bed made" is a recipe, not a dump
    (the 30B's talk bench: the answer was dropped)."""
    words = re.findall(r"[a-z_]+|\d+", line.lower().replace(" block", "_block"))
    counted = set()
    for i, w in enumerate(words):
        it = item_of(w, items)
        near = [int(x) for x in words[max(0, i - 1):i + 2] if x.isdigit()]
        if it and near and (pack is None or pack.get(it) in near):
            counted.add(it)
    return len(counted) >= 2
