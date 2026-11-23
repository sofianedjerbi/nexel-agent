"""One env in its own process, in the game's Python (3.14), spoken to over stdin/stdout by rl/vec.py (the trainer's
Python, with torch). Messages are length-prefixed pickles: ("spec",), ("reset", seed), ("step", action),
("close",). After a step that ends the episode, the trainer sends the next reset."""

import os
import pickle
import struct
import sys


def read(f):
    head = f.read(4)
    if len(head) < 4:
        return None
    return pickle.loads(f.read(struct.unpack("<I", head)[0]))


def write(f, obj):
    b = pickle.dumps(obj, protocol=5)
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def main():
    inp, out = sys.stdin.buffer, os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    sys.stdout = sys.stderr                    # (anything printed by the game goes to the log, not the pipe)
    from .core import SimEnv
    from .encode import N_ACTIONS, N_CHARS, SIDE, VEC, action_name
    env = SimEnv(**{k: int(v) for k, v in (a.split("=") for a in sys.argv[1:])})
    while (msg := read(inp)) is not None:
        if msg[0] == "spec":
            write(out, {"n_actions": N_ACTIONS, "n_chars": N_CHARS, "side": SIDE, "vec": VEC,
                        "names": [action_name(i) for i in range(N_ACTIONS)]})
        elif msg[0] == "reset":
            write(out, env.reset(msg[1]))
        elif msg[0] == "step":
            write(out, env.step(int(msg[1])))
        elif msg[0] == "close":
            break


if __name__ == "__main__":
    main()
