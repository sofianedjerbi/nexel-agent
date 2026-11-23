"""The player's own words: a chat reply when someone talks to it (with what it knows of them and what happened
between them, so it talks like someone who was there), being hit (told to the planner, never a scripted line), and
its diary: every so often one model call sums up what happened (memory.py). Mixed into Thinker via Planner.
"""

import asyncio
import json
import logging
import time

import httpx

from .fastprompt import chat_parse

log = logging.getLogger("thinker")


def chat_messages(name: str, persona: str, goal: str, speaker: str, text: str, recent=(), me=None, person="",
                  memories=(), said=()) -> list[dict]:
    """A player talks to you: answer them like a person in a game chat who remembers them (person: what you know of
    them; memories: what else matters; said: your own last lines, never said again)."""
    me = me or {}
    inv = ", ".join(f"{k} {v}" for k, v in sorted((me.get("inventory") or {}).items(), key=lambda kv: -kv[1])[:6]) or "nothing"
    sys_ = (f"You are {name}, a player in a survival sandbox game. Your character: {persona}\n"
            "Another player is talking to you in the local chat. Type like a real person in a game chat: casual, one "
            "short sentence (at most 20 words), about what they actually said and what is going on between you. Your "
            "character shows in your attitude, not in theatre. Answer questions truthfully from the facts given; never "
            "cryptic, no coordinates unless asked. If you talked before, carry on the conversation (no new hello). "
            "If they hurt you, you remember it and it shows. Never repeat a line you already said.\n"
            'Reply with JSON only: {"say": "<your line>", "do": "come" | "stop" | "nothing", "promise": "<what you '
            'agreed to do for them, or empty>"}. do=come: they ask you to come, follow or help them; do=stop: they ask '
            "you to stop or to do something else; otherwise nothing.")
    facts = [f"You are busy with: {goal or 'getting by'}. You are at ({me.get('x', '?')}, {me.get('y', '?')}), "
             f"health {me.get('hp', '?')}/20, food {me.get('hunger', '?')}/20, carrying: {inv}."]
    if person:
        facts.append("What you know of them: " + person)
    if memories:
        facts.append("You remember: " + " | ".join(memories))
    if recent:
        facts.append("Said around you just now: " + " | ".join(recent))
    if said:
        facts.append("Your own last lines (do not reuse): " + " | ".join(f'"{s}"' for s in said))
    return [{"role": "system", "content": sys_},
            {"role": "user", "content": "\n".join(facts) + f"\n{speaker} says: {text}\n/no_think"}]


def promise_of(out: str) -> str:
    try:
        d = json.loads(out[out.index("{"):out.rindex("}") + 1])
    except ValueError:
        return ""
    p = d.get("promise") if isinstance(d, dict) else ""
    return p.strip()[:80] if isinstance(p, str) and p.strip().lower() not in ("", "none", "nothing", "empty") else ""


class Chat:
    def on_hit(self, e, me):
        """Hit: the plan stops and the next one is made knowing it (who, with what, how hurt you are). What to do about
        it is the model's: hit back, run, talk. Not a scripted line."""
        who, dmg = str(e.get("attacker")), e.get("damage")
        b = next((x for x in (self.obs or {}).get("entities", []) if x.get("name") == who), None)
        tag = f" #{b['id']}" if b else ""
        self.heard.append(f"{who}{tag} just hit you with {e.get('weapon') or 'a bite'}: -{dmg} health, you have {me['hp']}/20")
        self.abort(f"{who} hit you")

    def on_chat(self, speaker, text, at=None):
        if self.chat_task and not self.chat_task.done():
            self.chat_next = (speaker, text, at)
        else:
            self.chat_task = asyncio.create_task(self.chat_reply(speaker, text, at))

    async def chat_reply(self, speaker, text, at=None):
        while True:     # (one line said back, and what was asked of you done: "come" walks to them, "stop" drops the plan)
            t0, me = time.monotonic(), (self.obs or {}).get("self", {})
            mem = self.mem
            said = mem.people.get(speaker, {}).get("you_said", [])[-3:]
            msgs = chat_messages(self.name, self.persona, self.goal or "", speaker, text, list(self.heard)[-3:-1], me,
                                 mem.person_text(speaker), mem.relevant(f"{text} {self.goal or ''}", (me.get("x"), me.get("y")),
                                                                        [speaker], k=3), said)
            try:
                r = await self.http.post(self.model_url, timeout=30, json=self._body(msgs, 80, 0.8, False))
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"] or ""
                (line, do), promise = chat_parse(out), promise_of(out)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                log.warning("%s: chat error %r", self.name, e)
                line, do, promise = "", "", ""
            if do in ("come", "stop"):
                self.abort(f"{speaker} asked you to {do}")
                self.history.append(f"({speaker} asked you to {do}: you did)")
            if do == "stop":
                self.history.append(f'({speaker} told you: "{text[:80]}". Do what they asked; pick another goal if it was about yours)')
            if promise:
                mem.promise(speaker, promise)
                self.history.append(f"(you promised {speaker}: {promise})")
            if line:
                mem.replied(speaker, line)
                self.queue.appendleft(self.item("say", {"text": line[:120]}, "chat", f"answering {speaker}"))
            if do == "come" and at:
                self.queue.insert(1 if line else 0, self.item("move_to", {"x": at[0], "y": at[1]}, "chat", f"going to {speaker}"))
            if line or do:
                self.new_plan.set()
                self.calls.append({"kind": "chat", "latency_s": round(time.monotonic() - t0, 2)})
                self.decision_log.write(json.dumps({"kind": "chat", "t": time.time(), "to": speaker, "heard": text,
                                                    "said": line, "do": do, "promise": promise,
                                                    "latency_s": round(time.monotonic() - t0, 2)}) + "\n")
            if not self.chat_next:
                return
            (speaker, text, at), self.chat_next = self.chat_next, None

    async def reflect(self, http):
        """The diary: when due, one model call sums up the recent events (and, the log full, merges the oldest)."""
        if not self.mem.summary_due():
            return
        msgs, n = self.mem.summary_request(self.name)
        t0 = time.monotonic()
        try:
            r = await http.post(self.model_url, timeout=60, json=self._body(msgs, 140, 0.3, False))
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log.warning("%s: diary error %r", self.name, e)
            return
        self.mem.summarized(text, n)
        self.decision_log.write(json.dumps({"kind": "diary", "t": time.time(), "covers": n, "text": text[:500],
                                            "latency_s": round(time.monotonic() - t0, 2)}) + "\n")
