"""A vision model's view of a build (#32, judge v3 part 2): its renders (tools/render_build.py: three quarters and
from over it) put to a model with one fixed rubric, neat, cosy and purposeful, each scored 1 to 5 with a reason. It is
used to check judge v3's rules (rl/judge.py looks), never alone, and calibrated on real players' builds first (#32).

The model is any server that speaks the OpenAI chat API with images (a local VLM: mlx_vlm.server, llama.cpp's
llama-server with an mmproj; or a hosted one): `url` is its base, e.g. http://127.0.0.1:8080/v1.

    from build_vlm import rubric
    rubric([top_png, three_quarter_png], "http://127.0.0.1:8080/v1")   # -> {"neat": {"score", "reason"}, ...}"""

import base64
import json
import re
from pathlib import Path

import httpx

CRITERIA = ("neat", "cosy", "purposeful")
# the rubric: fixed, so that scores are comparable across builds, runs and models (change it: a new rubric version)
RUBRIC_VERSION = 1
PROMPT = """You are judging a small house a player built in a cosy pixel-art farming game (top-down, tile-based: one
tile is one block). The pictures show the same build from high over it (its roof cut away, so you see inside) and as
a player sees it (three quarters, from the south).

Score it on three criteria, each from 1 (poor) to 5 (excellent):
- neat: straight walls, a clean rectangular footprint (or one made of rectangles), one wall material or a deliberate
  second one, a full roof, no stray or floating pieces around it.
- cosy: it feels like a home a person would want to be in: furnished (a bed, a table, a light, a chest), not empty,
  not cramped, warm.
- purposeful: it looks made on purpose: the door faces a path, the furniture is placed with sense, windows where they
  help, nothing random.

Judge only what the pictures show. Answer with JSON only, no other text, exactly in this form:
{"neat": {"score": 1-5, "reason": "one sentence"}, "cosy": {"score": 1-5, "reason": "one sentence"},
 "purposeful": {"score": 1-5, "reason": "one sentence"}}"""


def image_part(png: Path) -> dict:
    """A PNG as a chat message's image part (a data URL)."""
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(Path(png).read_bytes()).decode()}}


def request(pngs: list, model: str = "default") -> dict:
    """The chat request the rubric sends: the prompt and the renders, deterministic (temperature 0)."""
    return {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}] + [image_part(p) for p in pngs]}]}


def parse(text: str) -> dict:
    """The model's answer -> {criterion: {"score": 1..5, "reason": str}}; the first JSON object in it (some models wrap
    it in a code fence). Raises ValueError when a criterion is missing or out of range."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in the answer: {text[:200]!r}")
    got = json.loads(m.group(0))
    out = {}
    for c in CRITERIA:
        v = got.get(c) or {}
        score = int(v.get("score", 0))
        if not 1 <= score <= 5:
            raise ValueError(f"{c}: score {v.get('score')!r} not 1-5")
        out[c] = {"score": score, "reason": str(v.get("reason", "")).strip()}
    return out


def rubric(pngs: list, url: str, model: str = "default", timeout: float = 300) -> dict:
    """The rubric's scores of a build from its renders, by the model at `url` (an OpenAI-style /v1 base) ->
    {"neat", "cosy", "purposeful": {"score", "reason"}, "mean": float, "rubric": RUBRIC_VERSION, "model": model}."""
    r = httpx.post(url.rstrip("/") + "/chat/completions", json=request(pngs, model), timeout=timeout)
    r.raise_for_status()
    scores = parse(r.json()["choices"][0]["message"]["content"])
    return scores | {"mean": sum(s["score"] for s in scores.values()) / len(CRITERIA), "rubric": RUBRIC_VERSION, "model": model}
