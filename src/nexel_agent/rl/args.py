"""The live player's command line (rl/live.py main): the body, the game, the model, the run's notes."""

import argparse


def parser(policy, model_url) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nexel-agent", description="The AI player of Nexel Worlds: a trained body, a local model for its plans and words.")
    p.add_argument("--name", default="Rook")
    p.add_argument("--persona", default="a quiet player who learned the world by living in it; wants a house of its own with "
                   "a bed and a door, and good tools, and gets what it lacks from other players by asking or trading")
    p.add_argument("--policy", default=str(policy), help="the learned body (default: the shipped models/rook.npz)")
    p.add_argument("--game", default="ws://127.0.0.1:7777/agent")
    p.add_argument("--model-url", default=model_url, help="a local model server for the words and the planner (none: it does not "
                   "talk; if nothing answers there it plays on in silence); serve voice.MODEL there (Qwen3-30B-A3B-Instruct-2507 "
                   "4-bit), or voice.FALLBACK_MODEL")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--log", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask", type=int, default=1, help="the policy was trained with the legal-action mask")
    p.add_argument("--teleport", default="", help="x,y,z: on a --dev server, start there")
    p.add_argument("--brain", type=int, default=0, help="serve what it sees and thinks on 127.0.0.1:PORT/feed (rl/brain.py)")
    p.add_argument("--memory", default="", help="where the LLM layer keeps its notes on people (memory.json)")
    p.add_argument("--architect", type=int, default=1, help="a home not begun is drawn by the model first (#66)")
    return p
