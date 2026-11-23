"""The fast layer: the model's own trigger cards (cards.py) checked on every look, a few times a second, so a hit or a
being coming close is answered in a fraction of a second by what the model wrote, and the slow mind is told.
Card-writing calls: when the player joins, when a new kind of being first comes into view, after a hit that no card
answered. Mixed into Thinker.
"""

import asyncio
import json
import logging
import time

import httpx

from .cards import Cards, card_request, parse_cards
from .plans import parse_command
from .senses import rules_text
from .sim.entities import CREATURES

log = logging.getLogger("thinker")
SENSE_S = 0.2                                     # a look this often while an action runs


class React:
    def react_init(self, data_dir):
        self.cards = Cards(data_dir / "_cards.json", self.name)
        self.kinds_seen: set = set()
        self.card_call: asyncio.Task | None = None
        self.running: asyncio.Task | None = None
        self.firing: dict | None = None               # the card being played: for its latency line

    def on_look(self, obs, hits, t_seen):
        """After each look: new kinds of beings -> a card call; a card that holds -> it plays now."""
        kinds = {e["kind"] for e in obs["entities"] if e["kind"] in CREATURES or e["kind"] == "agent"}
        first = not self.kinds_seen
        new = kinds - self.kinds_seen
        self.kinds_seen |= kinds
        if first or new:
            self.kinds_seen.add("_joined")
            what = sorted(new) if not first else sorted(kinds | {k for k, c in CREATURES.items() if c.get("damage")} | {"agent"})
            self.ask_cards(f"{'You just joined the world.' if first else 'New in view: ' + ', '.join(sorted(new)) + '.'}",
                           what, "join" if first else "new_kind")
        m = self.cards.match(obs, hits)
        if m:
            self.fire(*m, obs, t_seen, hits)
        elif hits:
            who = ", ".join(sorted({h[0] for h in hits}))
            self.ask_cards(f"{who} just hit you; you have {obs['self']['hp']}/20 health. No card of yours answered.",
                           sorted({(h[1] or {}).get("kind", "agent") for h in hits}), "hit")

    def fire(self, i, card, cmds, obs, t_seen, hits):
        parsed = [c for c in map(parse_command, cmds) if c]
        if not parsed:
            return
        self.cards.fired_now(i)
        self.abort(f"your card fired: when {card['when']} do {card['do']}", tell=False)
        items = [self.item(*c, "card", f"card: {card['why'] or card['when']}") for c in parsed]   # (the new epoch)
        ages = [(obs["tick"] - e.get("tick", obs["tick"])) / 10 for e in obs["events"] if e.get("type") == "attack"]
        self.firing = {"t_seen": t_seen, "event_age_s": max(ages) if ages else 0.0, "card": card, "cmds": cmds,
                       "hit": bool(hits)}
        for it in reversed(items):
            self.queue.appendleft(it)
        if self.running and not self.running.done() and getattr(self, "current_source", "") != "card":
            self.running.cancel()                     # (the action on the server is replaced by the card's)
        self.new_plan.set()
        self.history.append(f"(your card fired: {card['do']} because {card['why'] or card['when']})")

    def card_sent(self, item):
        """The first command of a fired card goes out: its reaction time, logged."""
        f, self.firing = self.firing, None
        if not f:
            return
        rec = {"kind": "card", "t": time.time(), "do": f["cmds"], "when": f["card"]["when"], "author": f["card"].get("author"),
               "reaction_s": round(time.monotonic() - f["t_seen"] + f["event_age_s"], 3),
               "since_look_s": round(time.monotonic() - f["t_seen"], 3), "hit": f["hit"]}
        self.decision_log.write(json.dumps(rec) + "\n")
        log.info("%s [card %.2fs] %s", self.name, rec["reaction_s"], f["cmds"])

    def ask_cards(self, situation, kinds, source):
        if self.card_call and not self.card_call.done() or not getattr(self, "http", None):
            return
        self.card_call = asyncio.create_task(self.write_cards(situation, kinds, source))

    async def write_cards(self, situation, kinds, source):
        """One model call: the model writes its own reaction cards for what it may meet (its persona decides)."""
        pairs = [(k, CREATURES[k]["char"]) if k in CREATURES else (k, "P") for k in kinds]
        me = (self.obs or {}).get("self", {})
        msgs = card_request(self.name, self.persona, pairs, f"{situation} You: health {me.get('hp')}/20, food "
                            f"{me.get('hunger')}/20, carrying {', '.join(me.get('inventory', {})) or 'nothing'}.", rules_text())
        t0 = time.monotonic()
        try:
            r = await self.http.post(self.model_url, timeout=60, json=self._body(msgs, 300, 0.4, False))
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log.warning("%s: cards error %r", self.name, e)
            return
        new = self.cards.add(parse_cards(text), source)
        self.decision_log.write(json.dumps({"kind": "cards", "t": time.time(), "source": source, "situation": situation,
                                            "new": new, "raw": text[:600], "latency_s": round(time.monotonic() - t0, 2)}) + "\n")
        log.info("%s [cards %s] %d new: %s", self.name, source, len(new), [(c["when"], c["do"]) for c in new])

    async def sense_loop(self, game):
        """While an action runs, look anyway (cards are checked on every look)."""
        while True:
            await asyncio.sleep(SENSE_S)
            if self.running and not self.running.done():
                try:
                    await self.observe(game)
                except Exception as e:                    # noqa: BLE001  (a missed look is no reason to stop)
                    log.debug("sense: %r", e)
