#!/bin/sh
# Point this checkout's git at .githooks/ (commit-msg, pre-commit, pre-push: the checks CI runs). Once per checkout.
cd "$(dirname "$0")/.." && git config core.hooksPath .githooks && echo "hooks: $(git config core.hooksPath)"
