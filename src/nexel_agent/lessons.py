"""Lessons grounded in outcomes: what actually happened when the agent tried things, deduplicated."""

import json
from pathlib import Path


class Lessons:
    """Knowledge derived only from observed outcomes, deduplicated by key."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def learn(self, tool, args, reply):
        if reply.get("ok"):
            res = reply["result"]
            got = res.get("changed", {}).get("inventory", {})
            if tool == "dig" and got:
                tiles = res.get("changed", {}).get("tiles", [])
                self._add(f"dig:{','.join(sorted(got))}", f"Digging gave {', '.join(f'{v} {k}' for k, v in got.items())}"
                          + (f" (the tile became {tiles[0]['block'] if tiles[0]['block'] != 'none' else tiles[0]['ground']})"
                             if tiles else ""))
            elif tool == "craft":
                self._add(f"craft:{args.get('recipe')}", f"Crafting {args.get('recipe')} works: "
                          f"{', '.join(f'{v:+d} {k}' for k, v in got.items())}")
            elif tool == "eat":
                ch = res.get("changed", {})
                self._add(f"eat:{args.get('item')}", f"Eating {args.get('item')} changed hunger by {ch.get('hunger', 0):+d}")
        else:
            code = reply.get("error")
            if code in ("interrupted", "rate_limited", None):
                return
            key = f"fail:{tool}:{code}:{args.get('recipe') or args.get('item') or ''}"
            self._add(key, f"{tool} {json.dumps(args)} failed: {code} ({reply.get('detail')})")

    def _add(self, key, text):
        if self.items.get(key) != text:
            self.items[key] = text
            self.path.write_text(json.dumps(self.items, indent=1))

    def text(self, limit=25) -> str:
        return "\n".join(f"- {v}" for v in list(self.items.values())[-limit:]) or "- (nothing learned yet)"
