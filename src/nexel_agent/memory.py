"""A player's memory, kept in agents/<name>/memory.json (it survives restarts), managed by the model itself.

- episodes: what happened that is worth keeping (who said or did what, fights, deaths and their cause, places found),
  each with the time, the game clock and where. The harness writes these from what it observed: facts, not guesses.
- notes: what the model chose to remember (`remember <text>`) and can drop (`forget <n>`).
- people: per person, what they said lately, what you said back, hits both ways, gifts, promises.
- summaries: the model's own short account of a stretch of play (written every SUMMARY_S, or when the log is full:
  then the oldest episodes are merged into one summary by the model and dropped: consolidation).

Retrieval for a prompt: the few entries that matter now, scored by a person named or in view, a place near you,
words shared with the goal, and recency.
"""

import json
import re
import time
from pathlib import Path

MAX_EPISODES = 60                 # past this, the oldest MERGE are merged by the model into a summary
MERGE = 30
MAX_NOTES = 20
MAX_SUMMARIES = 8
SUMMARY_S = 600.0                 # a summary of the last stretch every 10 min of play
HALF_LIFE_S = 900.0
STOP = set("the a an to of and in on at for with my your is it you get go make find some more".split())


def words(text: str) -> set:
    return {w for w in re.findall(r"[a-z_]+", (text or "").lower()) if w not in STOP and len(w) > 2}


class Memory:
    def __init__(self, path: Path | None = None, now=time.time):
        self.path, self.now = path, now
        d = json.loads(path.read_text()) if path and path.exists() else {}
        self.episodes: list = d.get("episodes", [])
        self.notes: list = d.get("notes", [])
        self.people: dict = d.get("people", {})
        self.summaries: list = d.get("summaries", [])
        self.next_id: int = d.get("next_id", 1)
        self.last_summary: float = d.get("last_summary", self.now())

    def save(self):
        if self.path:
            self.path.write_text(json.dumps({"episodes": self.episodes, "notes": self.notes, "people": self.people,
                                             "summaries": self.summaries, "next_id": self.next_id,
                                             "last_summary": self.last_summary}))

    # ------------------------------------------------------------ writing
    def add(self, kind: str, text: str, who: str | None = None, at=None, when: str = ""):
        """An episode. The same text twice in a row is kept once (a hit every second is one fight, counted)."""
        last = self.episodes[-1] if self.episodes else None
        if last and last["text"] == text and last["kind"] == kind:
            last["n"], last["t"] = last.get("n", 1) + 1, self.now()
        else:
            self.episodes.append({"t": self.now(), "when": when, "at": list(at) if at else None, "kind": kind,
                                  "who": who, "text": text[:160]})
        self.save()

    def person(self, name: str) -> dict:
        return self.people.setdefault(name, {"first": self.now(), "said": [], "you_said": [], "hit_you": 0,
                                             "you_hit": 0, "gave_you": {}, "promised": []})

    def heard(self, name, text, at=None, when=""):
        p = self.person(name)
        p["said"] = (p["said"] + [text[:100]])[-4:]
        p["last"] = self.now()
        self.add("talk", f'{name} said: "{text[:100]}"', name, at, when)

    def replied(self, name, text):
        p = self.person(name)
        p["you_said"] = (p["you_said"] + [text[:100]])[-4:]
        self.save()

    def promise(self, name, text):
        p = self.person(name)
        p["promised"] = (p["promised"] + [text[:80]])[-3:]
        self.save()

    def hit_by(self, name, damage, weapon, at=None, when="", hp=None):
        if not name.startswith("a "):                          # a creature ("a tusker") is no person
            self.person(name)["hit_you"] += 1
        self.add("fight", f"{name} hit you with {weapon or 'a bite'} (-{damage})" + (f", you at {hp}/20" if hp is not None else ""),
                 name, at, when)

    def you_hit(self, name):
        self.person(name)["you_hit"] += 1
        self.save()

    def remember(self, text: str, at=None, when="") -> int:
        n = self.next_id
        self.next_id += 1
        self.notes = (self.notes + [{"id": n, "t": self.now(), "at": list(at) if at else None, "when": when,
                                     "text": text.strip()[:140]}])[-MAX_NOTES:]
        self.save()
        return n

    def forget(self, n: int) -> bool:
        before = len(self.notes)
        self.notes = [x for x in self.notes if x["id"] != n]
        self.save()
        return len(self.notes) < before

    # ------------------------------------------------------------ reading
    def _score(self, e, names, at, goal_words, now):
        s = 0.5 ** ((now - e["t"]) / HALF_LIFE_S)
        text = e["text"].lower()
        if any(n.lower() in text or n == e.get("who") for n in names):
            s += 2.0
        if at and e.get("at") and abs(e["at"][0] - at[0]) + abs(e["at"][1] - at[1]) <= 15:
            s += 1.0
        s += 0.7 * len(goal_words & words(text))
        if e.get("kind") in ("death", "fight"):
            s += 0.5
        return s

    def relevant(self, goal="", at=None, names=(), k=4) -> list[str]:
        """The k episodes that matter now, oldest first, as short lines."""
        now, gw = self.now(), words(goal)
        top = sorted(self.episodes, key=lambda e: -self._score(e, names, at, gw, now))[:k]
        return [self.line(e, now) for e in sorted(top, key=lambda e: e["t"])]

    def line(self, e, now=None) -> str:
        ago = int(((now or self.now()) - e["t"]) // 60)
        where = f" at ({e['at'][0]},{e['at'][1]})" if e.get("at") else ""
        times = f" (x{e['n']})" if e.get("n", 1) > 1 else ""
        return f"{ago} min ago{where}: {e['text']}{times}"

    def notes_text(self) -> str:
        return " ".join(f"[{x['id']}] {x['text']}" + (f" ({x['at'][0]},{x['at'][1]})" if x.get("at") else "")
                        for x in self.notes[-6:])

    def person_text(self, name) -> str:
        """What you know of someone, for a chat with them or a plan near them."""
        p = self.people.get(name)
        if not p:
            return f"You have not met {name} before."
        bits = [f"met {int((self.now() - p['first']) // 60)} min ago"]
        if p["said"]:
            bits.append("they said: " + " / ".join(f'"{s}"' for s in p["said"][-3:]))
        if p["you_said"]:
            bits.append("you said: " + " / ".join(f'"{s}"' for s in p["you_said"][-3:]))
        if p["hit_you"]:
            bits.append(f"they hit you {p['hit_you']}x")
        if p["you_hit"]:
            bits.append(f"you hit them {p['you_hit']}x")
        if p["gave_you"]:
            bits.append("they gave you " + ", ".join(f"{n} {k}" for k, n in p["gave_you"].items()))
        if p["promised"]:
            bits.append("you promised: " + "; ".join(p["promised"]))
        return f"{name}: " + "; ".join(bits) + "."

    def summary_text(self) -> str:
        return self.summaries[-1]["text"] if self.summaries else ""

    # ------------------------------------------------------------ the model's own summaries
    def summary_due(self) -> bool:
        return len(self.episodes) > MAX_EPISODES or (self.now() - self.last_summary >= SUMMARY_S and self.episodes)

    def summary_request(self, name) -> tuple[list[dict], int]:
        """(messages, how many of the oldest episodes it covers) for one model call."""
        n = MERGE if len(self.episodes) > MAX_EPISODES else len(self.episodes)
        lines = [self.line(e) for e in self.episodes[:n]]
        prev = self.summary_text()
        msgs = [{"role": "system", "content": f"You are {name}, a player in a survival game, keeping your own diary. "
                 "Write what you want to remember from these events: people (what they did, said, asked, what you "
                 "promised), places (home, workbench, chests, caves, water, dangers, with coordinates), fights and "
                 "deaths and their causes, what worked. Plain short sentences, at most 70 words. No preamble."},
                {"role": "user", "content": (f"Your diary so far: {prev}\n" if prev else "") + "Events:\n"
                 + "\n".join(lines) + "\n/no_think"}]
        return msgs, n

    def summarized(self, text: str, n: int):
        """The model's summary in; the n episodes it covers out when the log was full (consolidated)."""
        text = " ".join(text.split())[:500]
        if text:
            self.summaries = (self.summaries + [{"t": self.now(), "text": text}])[-MAX_SUMMARIES:]
            if len(self.episodes) > MAX_EPISODES:
                self.episodes = self.episodes[n:]
        self.last_summary = self.now()
        self.save()

