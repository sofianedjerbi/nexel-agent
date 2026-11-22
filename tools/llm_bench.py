"""Time the thinker's model calls on real observations: prompt tokens, cached tokens, prefill (time to the
first token) and decode (the rest), for the plan prompt and the chat reply.

    PYTHONPATH=src python tools/llm_bench.py --game ws://127.0.0.1:7843/agent --model-url http://127.0.0.1:8143/v1/chat/completions --n 6
"""

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx

from megastructure.client import Client
from megastructure.fastprompt import chat_messages
from megastructure.plans import parse_plan
from megastructure.thinker import Thinker


async def timed(http, url, messages, max_tokens, temperature):
    body = {"model": "default_model", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.monotonic()
    first, text, usage, n = None, "", {}, 0
    async with http.stream("POST", url, json=body, timeout=300) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    if first is None:
                        first = time.monotonic()
                    text += piece
                    n += 1
    end = time.monotonic()
    first = first or end
    return {"total": end - t0, "prefill": first - t0, "decode": end - first, "text": text, "usage": usage, "chunks": n}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ws://127.0.0.1:7843/agent")
    p.add_argument("--model-url", default="http://127.0.0.1:8143/v1/chat/completions")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--mode", choices=["plan", "chat"], default="plan")
    a = p.parse_args()
    t = Thinker("bench", "a curious scavenger who wants to survive and build a home", data_dir=Path(tempfile.mkdtemp()),
                game_url=a.game, model_url=a.model_url)
    rows = []
    async with httpx.AsyncClient() as http, Client(a.game) as game:
        await game.hello("bench", "thinker-bench")
        for i in range(a.n):
            obs = await game.observe()
            if a.mode == "chat":
                msgs, mt, temp = chat_messages(t.name, t.persona, "get wood", "Ana", "hi! got any wood to spare?"), 40, 0.6
            else:
                msgs, mt, temp = t.plan_messages(obs), t.max_tokens, t.temperature
            r = await timed(http, a.model_url, msgs, mt, temp)
            u = r["usage"]
            ok = bool(parse_plan(r["text"])[0]) if a.mode == "plan" else bool(r["text"].strip())
            rows.append(r)
            print(f"{i}: prompt {u.get('prompt_tokens')} cached {u.get('prompt_tokens_details', {}).get('cached_tokens')} "
                  f"out {u.get('completion_tokens')} | prefill {r['prefill']:.2f}s decode {r['decode']:.2f}s "
                  f"total {r['total']:.2f}s valid={ok} | {r['text'][:160]!r}", flush=True)
            if a.mode == "plan":
                for tool, args in parse_plan(r["text"])[0]:
                    await game.act(tool, **args)
    tot = sorted(r["total"] for r in rows)
    print(f"total p50 {statistics.median(tot):.2f}s max {tot[-1]:.2f}s; prefill p50 "
          f"{statistics.median(r['prefill'] for r in rows):.2f}s; decode p50 {statistics.median(r['decode'] for r in rows):.2f}s")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
