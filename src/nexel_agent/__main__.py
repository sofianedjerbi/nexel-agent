"""`python -m nexel_agent` = `nexel-agent`: the trained player (rl/live.py, models/rook.npz)."""

from .rl.live import main

if __name__ == "__main__":       # (importing this file never starts a player)
    main()
