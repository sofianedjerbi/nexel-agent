"""How long a reply waits for the model while the planner thinks (#11): rl/voice.py Voice.post, against a fake model
server that takes as long as the 30B (a reply ~1 s, a thought 2-4 s: rl_runs/bench).
A planner thinks again and again (the worst case: always thinking); lines come to answer at random. It prints the
replies' wait (queue) and their whole time, p50 and p90, and how many thoughts were made and given up.

    python tools/reply_wait.py [--seconds 120] [--serial]

--serial: a server that answers one request at a time (without it: one that decodes those in hand together, as
mlx-lm 0.32.2's does). Times run 10x faster than said and are reported unscaled."""

import argparse
import asyncio
import random
import statistics

from nexel_agent.rl.voice import THOUGHT_TOKENS, WORDS_TOKENS, Voice

SCALE = 0.1                  # (the run's clock: 10x faster than the times it reports)


class SlowServer:
    """As long as the 30B takes. batch (mlx-lm 0.32.2's server, as players.py starts it: --decode-concurrency 32):
    the requests in hand are decoded together, each a little slower for the others (BATCH a request more);
    otherwise one at a time."""
    BATCH = .25

    def __init__(self, batch: bool, rng: random.Random):
        self.slot, self.batch, self.rng, self.left = asyncio.Lock(), batch, rng, {}

    async def post(self, url, timeout=None, json=None):
        took = (1.0 + self.rng.uniform(-.2, .2)) if json["max_tokens"] <= WORDS_TOKENS else self.rng.uniform(2.0, 4.0)
        if not self.batch:
            async with self.slot:
                await asyncio.sleep(took * SCALE)
            return Reply()
        me = object()
        self.left[me] = took
        try:
            while self.left[me] > 0:              # (a step of decoding for every request in hand)
                await asyncio.sleep(.01 * SCALE)
                self.left[me] -= .01 / (1 + self.BATCH * (len(self.left) - 1))
        finally:
            del self.left[me]
        return Reply()


class Reply:
    def json(self):
        return {"choices": [{"message": {"content": "{}"}}], "usage": {"completion_tokens": 10}}


async def run(seconds: float, batch: bool, seed: int = 1) -> dict:
    rng = random.Random(seed)
    v = Voice("Rook", "a builder", "http://fake", SlowServer(batch, rng), log="/dev/null")
    replies, thoughts = [], {"made": 0, "given_up": 0}
    end = asyncio.get_running_loop().time() + seconds * SCALE

    async def planner():
        while asyncio.get_running_loop().time() < end:
            try:
                await v.post([], THOUGHT_TOKENS, words=False)
                thoughts["made"] += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                thoughts["given_up"] += 1
            await asyncio.sleep(.2 * SCALE)

    async def talk():
        while asyncio.get_running_loop().time() < end:
            await asyncio.sleep(rng.expovariate(1 / 4.0) * SCALE)      # (a line to answer every 4 s or so)
            t0 = asyncio.get_running_loop().time()
            _, t = await v.post([], WORDS_TOKENS, words=True)
            replies.append(((asyncio.get_running_loop().time() - t0) / SCALE, t["queue_s"] / SCALE))

    p = asyncio.ensure_future(planner())
    await talk()
    p.cancel()
    q = lambda xs, k: round(statistics.quantiles(xs, n=10)[k - 1] if len(xs) > 1 else xs[0], 2)    # noqa: E731
    total, wait = [r[0] for r in replies], [r[1] for r in replies]
    return {"replies": len(replies), "wait_p50": q(wait, 5), "wait_p90": q(wait, 9), "total_p50": q(total, 5),
            "total_p90": q(total, 9), **thoughts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--serial", action="store_true", help="a server that answers one request at a time")
    a = ap.parse_args()
    print(asyncio.run(run(a.seconds, not a.serial)))


if __name__ == "__main__":
    main()
