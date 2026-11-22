"""The AI players launcher: the roster and the child processes' lifecycle (no real processes)."""

import re
import subprocess

import pytest

from megastructure import players


class FakeProc:
    def __init__(self, cmd, hang=False, **kw):
        self.cmd, self.hang, self.returncode = cmd, hang, None
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.hang:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.returncode


def test_roster_is_big_unique_and_complete():
    roster = players.load_roster()
    assert len(roster) >= 40
    names = [p["name"] for p in roster]
    assert len(set(names)) == len(names)
    for p in roster:
        assert set(p) >= {"name", "role", "ref", "persona"}
        assert re.fullmatch(r"[a-z]+[0-9]", p["name"]), p["name"]   # lowercase reference + a digit
        assert len(p["name"]) <= 24                                   # the server's name limit
        assert 20 <= len(p["persona"]) <= 200                         # hello keeps 200 chars
        assert p["persona"] != "human"                                # a bot: robot voice, own race
    assert len({p["persona"] for p in roster}) == len(roster)
    assert len({p["role"] for p in roster}) >= 30


def test_arg_defaults_and_choice():
    a = players.parse_args([])
    assert (a.n, a.port, a.game, a.minutes) == (2, 8080, "ws://127.0.0.1:7777/agent", None)
    assert players.model_name(a.model) == "mlx-community/Qwen3-8B-4bit"
    assert players.model_name("4b") == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    roster = players.load_roster()
    assert len({p["name"] for p in players.choose(roster, 6, None, seed=1)}) == 6
    assert [p["name"] for p in players.choose(roster, 4, "senku0,gatsu0", None)] == ["senku0", "gatsu0"]
    with pytest.raises(SystemExit):
        players.choose(roster, 4, "nobody9", None)
    with pytest.raises(SystemExit):
        players.choose(roster, 0, None, None)


def test_game_url_and_log_parsing():
    assert players.game_http("ws://127.0.0.1:7781/agent") == "http://127.0.0.1:7781/"
    assert players.game_http("wss://x.org/agent") == "https://x.org/"
    log = ("2026-11-22 19:40:01,000 gatsu0 joined\n"
           "2026-11-22 19:40:05,000 gatsu0 [model 3.2s] get wood | dig N: ok, +1 wood\n")
    assert players.last_event(log) == "model 3.2s: dig N: ok, +1 wood"
    assert players.last_event(log.splitlines()[0]) == "joined"
    assert players.last_event("") is None


def test_thinker_command_carries_persona_and_model():
    a = players.parse_args(["--minutes", "1", "--port", "8099"])
    p = players.load_roster()[0]
    cmd = players.Launcher(a).thinker_cmd(p)
    assert cmd[cmd.index("--persona") + 1] == p["persona"]
    assert cmd[cmd.index("--model") + 1] == "default_model"
    assert "http://127.0.0.1:8099/v1/chat/completions" in cmd and cmd[-2:] == ["--minutes", "1.0"]


def test_refuses_without_game_server(tmp_path, monkeypatch):
    monkeypatch.setattr(players, "reachable", lambda url: False)
    said = []
    a = players.parse_args(["--data", str(tmp_path / "agents")])
    assert players.Launcher(a, popen=FakeProc, out=said.append).run(players.load_roster()[:2]) == 2
    assert "not answering" in said[0]


def test_stop_kills_every_child_and_its_own_model_server(tmp_path, monkeypatch):
    made = []

    def popen(cmd, **kw):
        made.append(FakeProc(cmd, hang="mlx_lm.server" in cmd or cmd[cmd.index("--name") + 1] == "senku0"))
        return made[-1]

    ups = iter([False, True])
    monkeypatch.setattr(players, "model_up", lambda port: next(ups, True))
    monkeypatch.setattr(players, "reachable", lambda url: True)
    monkeypatch.setattr(players.Launcher, "warm", lambda self: None)
    monkeypatch.setattr(players.time, "sleep", lambda s: None)

    a = players.parse_args(["--data", str(tmp_path), "--stagger", "0"])
    ln = players.Launcher(a, popen=popen, out=lambda s: None)
    def alive():                          # interrupt once the players run, as Ctrl-C would
        raise KeyboardInterrupt

    ln.alive = alive
    assert ln.run(players.choose(players.load_roster(), 2, "gatsu0,senku0")) == 130
    assert len(made) == 3 and "mlx_lm.server" in made[0].cmd
    assert all(p.terminated for p in made)
    assert all(p.returncode is not None for p in made)               # nothing left running
    assert made[0].killed and made[2].killed and not made[1].killed  # the hung ones were killed


def test_reused_model_server_is_left_alone(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(players, "model_up", lambda port: True)
    monkeypatch.setattr(players, "reachable", lambda url: True)
    a = players.parse_args(["--data", str(tmp_path), "--pick", "luffy0"])
    ln = players.Launcher(a, popen=lambda cmd, **kw: made.append(FakeProc(cmd)) or made[-1], out=lambda s: None)
    ln._sleep_watching = lambda s: [p.terminate() for p in made]      # the player finishes
    assert ln.run(players.choose(players.load_roster(), 1, "luffy0")) == 0
    assert len(made) == 1 and "megastructure.thinker" in made[0].cmd
