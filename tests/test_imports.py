"""Every module of the package imports (the trainer's, which need torch and stable-baselines3 in their own env, when
those are installed), and nothing names a module that is gone."""

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import nexel_agent

ROOT = Path(__file__).resolve().parents[1]
TRAINER_ONLY = {"torch", "stable_baselines3", "sb3_contrib", "gymnasium"}
MODULES = sorted(m.name for m in pkgutil.walk_packages(nexel_agent.__path__, "nexel_agent."))


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name, monkeypatch):
    """(Importing must start nothing: a player that started here would join the game on this machine.)"""
    monkeypatch.setattr("asyncio.run", lambda *a, **k: pytest.fail(f"importing {name} runs a program"))
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] not in TRAINER_ONLY:
            raise
        pytest.skip(f"needs {e.name} (the trainer's env)")


def test_the_live_player_and_the_launcher_are_there():
    assert {"nexel_agent.rl.live", "nexel_agent.rl.mind", "nexel_agent.rl.voice", "nexel_agent.players"} <= set(MODULES)


def test_no_text_names_a_module_that_is_gone():
    """Each dotted module path written in the sources, tools, docs and workflows all exist."""
    texts = [p for d in ("src", "tests", "tools", "docs", ".github") for p in (ROOT / d).rglob("*")
             if p.suffix in (".py", ".sh", ".md", ".yml", ".toml")] + [ROOT / n for n in ("README.md", "CONTRIBUTING.md", "pyproject.toml")]
    gone = {(str(p.relative_to(ROOT)), m) for p in texts for m in re.findall(r"nexel_agent(?:\.[a-z][a-z_0-9]*)+", p.read_text())
            if m not in MODULES and m.rsplit(".", 1)[0] not in MODULES}
    assert not gone
