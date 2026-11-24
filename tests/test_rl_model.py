"""The shipped body (models/rook.npz) loads and acts on a real observation, with the legal-action mask."""

from pathlib import Path

import numpy as np
import pytest

from nexel_agent.rl.core import SimEnv
from nexel_agent.rl.encode import N_ACTIONS
from nexel_agent.rl.live import POLICY
from nexel_agent.rl.npolicy import NumpyPolicy


@pytest.mark.skipif(not Path(POLICY).exists(), reason="no shipped model in this checkout")
def test_the_shipped_body_acts():
    assert Path(POLICY).stat().st_size < 10 * 2 ** 20
    pol = NumpyPolicy(POLICY)
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    o = env.reset(7)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(30):
        i = pol.act(o, rng, o["mask"])
        assert 0 <= i < N_ACTIONS and o["mask"][i]                  # (a legal action, always)
        seen.add(i)
        o, *_ = env.step(i)
    assert len(seen) > 1
    from nexel_agent.rl.encode import V3
    marked = {**o, "vec": o["vec"].copy()}
    marked["vec"][V3:] = 1                                       # (inputs it was not trained on: it does not read them)
    assert np.array_equal(pol.logits(o), pol.logits(marked))
