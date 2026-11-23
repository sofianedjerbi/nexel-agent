"""The plan the real-time thinker gets back: {"goal": ..., "do": ["dig N axe", "move_to 4 -2", ...], "note": ...}.

A command is a tool name and its arguments in TOOL_SPECS order ("place E workbench", "give 7 wood 2"), or
key=value ("dig dir=N with=axe"). PlanStream hands out each command as soon as its string closes, so the
first action starts while the model is still writing the rest.
"""

import json
import re

from .sim.specs import TOOL_SPECS

DIR_WORDS = {"N", "E", "S", "W", "up"}
LOCAL_SPECS = {"remember": {"args": {"text": "str"}}, "forget": {"args": {"n": "int"}}, "drop_card": {"args": {"n": "int"}},
               "skill": {"args": {"text": "str"}}}   # the player's own memory
from .verbs import VERB_SPECS  # noqa: E402
SPECS = {**TOOL_SPECS, **LOCAL_SPECS, **VERB_SPECS}
FILLER_WORDS = {"with", "the", "a", "to", "at", "using", "x", "y", "dir", "facing"}   # "dig N with pickaxe", "move_to x 3 y 4"


def _kind(key: str, spec: str) -> str:
    if spec.startswith("N|E|S|W"):
        return "dir"
    if spec.startswith("int") or spec.endswith(" id"):
        return "int"
    if spec.startswith("bool"):
        return "bool"
    return "text" if key == "text" else "str"


def _optional(spec: str) -> bool:
    return "optional" in spec or "omit" in spec


def parse_command(item) -> tuple[str, dict] | None:
    """'dig N axe' / 'move_to 3 -4' / 'say hello there' / {"tool":..,"args":..} -> (tool, args), or None."""
    if isinstance(item, dict):
        tool = item.get("tool")
        args = item.get("args") if isinstance(item.get("args"), dict) else {k: v for k, v in item.items() if k != "tool"}
        return (tool, args) if tool in SPECS else None
    if not isinstance(item, str):
        return None
    words = item.strip().split()
    if not words:
        return None
    tool = words[0].lower()
    spec = SPECS.get(tool)
    if spec is None:
        return None
    keys = list(spec["args"])
    args: dict = {}
    rest = []
    for w in words[1:]:
        k, eq, v = w.partition("=")
        if eq and k in spec["args"]:
            args[k] = v.strip("\"'")
        elif w.lower() not in FILLER_WORDS or "text" in spec["args"]:
            rest.append(w)
    free = [k for k in keys if k not in args]
    d = next((i for i, w in enumerate(rest) if w.upper() in ("N", "E", "S", "W")), None)
    if free and _kind(free[0], spec["args"][free[0]]) == "dir" and d and "text" not in keys[:2]:
        rest.insert(0, rest.pop(d))               # "place workbench S" means "place S workbench"
    i = 0
    while i < len(rest) and free:
        k = free[0]
        kind = _kind(k, spec["args"][k])
        w = rest[i]
        if kind == "text":
            args[k] = " ".join(rest[i:]).strip("\"'")
            i = len(rest)
        elif kind == "dir":
            if w.upper() in ("N", "E", "S", "W") or w.lower() == "up":
                args[k] = "up" if w.lower() == "up" else w.upper()
                i += 1
        elif kind == "int":
            if re.fullmatch(r"[-+]?\d+", w.strip("#(),")):
                args[k] = int(w.strip("#(),"))
                i += 1
        elif kind == "bool":
            if w.lower() in ("run", "true", "list", "yes"):
                args[k] = True
                i += 1
        else:
            args[k] = w.strip("\"'")
            i += 1
        free.pop(0)
    for k, v in list(args.items()):              # key=value ints and bools
        kind = _kind(k, spec["args"][k])
        if kind == "int" and isinstance(v, str):
            try:
                args[k] = int(v)
            except ValueError:
                return None
        if kind == "bool" and isinstance(v, str):
            args[k] = v.lower() in ("1", "true", "yes", "run")
        if kind == "dir" and isinstance(v, str) and v not in DIR_WORDS:
            args[k] = v.upper()
    need = [k for k in keys if not _optional(spec["args"][k]) and k not in args and k not in ("with",)]
    if any(_kind(k, spec["args"][k]) in ("dir", "int") for k in need) and tool not in ("take",):
        return None                               # a command missing its direction or a number is not a command
    return tool, args


class PlanStream:
    """Feed the model's text as it streams; get each finished command the moment its string closes."""

    STR = re.compile(r'\s*("(?:[^"\\]|\\.)*")\s*([,\]])')

    def __init__(self):
        self.text = ""
        self.sent = 0
        self.goal = None
        self.note = None
        self.done_when = None
        self.repeat = 0

    def feed(self, piece: str) -> list:
        self.text += piece
        if self.goal is None:
            m = re.search(r'"goal"\s*:\s*("(?:[^"\\]|\\.)*")', self.text)
            if m:
                self.goal = json.loads(m.group(1))
        m = re.search(r'"do"\s*:\s*\[', self.text)
        if not m:
            return []
        items, pos = [], m.end()
        while True:
            s = self.STR.match(self.text, pos)
            if s:
                items.append(json.loads(s.group(1)))
                pos = s.end()
                if s.group(2) == "]":
                    break
                continue
            obj = _object_at(self.text, pos)
            if obj is None:
                break
            items.append(obj[0])
            pos = obj[1]
            t = re.match(r"\s*([,\]])", self.text[pos:])
            if not t:
                break
            pos += t.end()
            if t.group(1) == "]":
                break
        new = items[self.sent:]
        self.sent = len(items)
        return new

    def finish(self) -> list:
        """The rest at the end of the stream (a plan whose JSON is whole but oddly spaced), and the note."""
        d = _first_object(self.text)
        if isinstance(d, dict):
            self.goal = self.goal or (d.get("goal") if isinstance(d.get("goal"), str) else None)
            self.note = d.get("note") if isinstance(d.get("note"), str) else None
            self.done_when = d.get("done_when") if isinstance(d.get("done_when"), dict) else None
            r = d.get("repeat")
            self.repeat = (5 if r is True else int(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else 0)
            todo = d.get("do") if isinstance(d.get("do"), list) else ([d] if "tool" in d else [])
            new = todo[self.sent:]
            self.sent = max(self.sent, len(todo))
            return new
        return []


def _object_at(text, pos):
    i = len(text) - len(text[pos:].lstrip())
    if i >= len(text) or text[i] != "{":
        return None
    depth, instr, esc = 0, False, False
    for j in range(i, len(text)):
        c = text[j]
        if instr:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1]), j + 1
                except json.JSONDecodeError:
                    return None
    return None


def _first_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    i = text.find("{")
    while i != -1:
        o = _object_at(text, i)
        if o:
            return o[0]
        i = text.find("{", i + 1)
    return None


def parse_plan(text: str) -> tuple[list[tuple[str, dict]], str | None, str | None]:
    """A whole reply -> (commands, goal, note). Bad commands are dropped; none at all -> []."""
    ps = PlanStream()
    raw = ps.feed(text) + ps.finish()
    cmds = [c for c in (parse_command(x) for x in raw) if c]
    return cmds, ps.goal, ps.note


def command_text(tool: str, args: dict) -> str:
    return " ".join([tool] + [str(v) if not isinstance(v, bool) else k for k, v in args.items() if v is not False])
