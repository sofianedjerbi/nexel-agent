# Changelog

Written by [python-semantic-release](https://python-semantic-release.readthedocs.io) from the
[Conventional Commits](https://www.conventionalcommits.org) on main (see CONTRIBUTING.md).

<!-- version list -->

## v0.9.0 (2026-11-24)

### Features

- **rl**: The observation shows what it has already done this life
  ([`f3a2d72`](https://github.com/sofianedjerbi/nexel-agent/commit/f3a2d72c5ec5c980b6b49751f5b44df130fff0db))


## v0.8.0 (2026-11-24)

### Features

- **models**: Ship rook from r35 at +300k (score 5.63 to 5.83, rooms 2% to 5%)
  ([`8f022c2`](https://github.com/sofianedjerbi/nexel-agent/commit/8f022c27d4fb494abfb546aee8d3ce182a8fa33f))

- **rl**: Log the chance the policy gives a tool craft where one is possible
  ([`d46888b`](https://github.com/sofianedjerbi/nexel-agent/commit/d46888bbd4e6d118dd080f40aa6b4fa93bbedf7d))


## v0.7.0 (2026-11-24)

### Features

- **models**: Ship rook from r34 at +900k (deaths 4.0 to 1.0, first strikes 0.10 to 0.07)
  ([`551ef0a`](https://github.com/sofianedjerbi/nexel-agent/commit/551ef0ab97da2c97f2335ea773c463fc378bcbc9))

- **rl**: Starts one step from a tool, and gear numbers in the checks
  ([`810afd3`](https://github.com/sofianedjerbi/nexel-agent/commit/810afd318fa0f058740d7b053141749410393349))


## v0.6.0 (2026-11-24)

### Features

- **models**: Ship rook from r34 (first strikes per near-minute 0.24 to 0.10)
  ([`d7af322`](https://github.com/sofianedjerbi/nexel-agent/commit/d7af322b15ad218265f3dd9a9d58c55b5cfcfdf5))


## v0.5.1 (2026-11-24)

### Bug Fixes

- **rl**: Close three reward holes behind first strikes
  ([`0b91419`](https://github.com/sofianedjerbi/nexel-agent/commit/0b9141988b7c44c612bc90fac4bf79127e0d9a3d))


## v0.5.0 (2026-11-23)

### Features

- **rl**: Ring places what the pack holds, fill and stock ladders, the attack mask uses the server's
  reach
  ([`b64c499`](https://github.com/sofianedjerbi/nexel-agent/commit/b64c4999657ceee4a1d5e1e5bc5b8847b82d0c0d))

### Refactoring

- **agent**: Retire the thinker generation, nexel-agent runs the trained player
  ([`2a21e63`](https://github.com/sofianedjerbi/nexel-agent/commit/2a21e634b7776b8fa6f068908281d7c8d91c572b))

- **rl**: One reader for the game's events and one distance (rl/facts.py)
  ([`44524cf`](https://github.com/sofianedjerbi/nexel-agent/commit/44524cf31fa044794501a8fa7ffbb420624045e7))


## v0.4.1 (2026-11-23)

### Bug Fixes

- **voice**: One situation text for the words and the planner, one reply per burst
  ([`c647e2f`](https://github.com/sofianedjerbi/nexel-agent/commit/c647e2f4ea116753d42905036173eb09fccec97a))


## v0.4.0 (2026-11-23)

### Features

- **rl**: Ship Rook's trained body in models/rook.npz, the default policy
  ([`b4df5fd`](https://github.com/sofianedjerbi/nexel-agent/commit/b4df5fd635979e9d60c2c3abc586f7d5dfa98645))


## v0.3.0 (2026-11-23)

### Features

- **rl**: Ring, one decision that walls the player in when the pack can close a room
  ([`560d103`](https://github.com/sofianedjerbi/nexel-agent/commit/560d1035f1b2fa4a005a28502b9aa0e52d190502))


## v0.2.0 (2026-11-23)

### Documentation

- Readme is one animated banner, a line and the links
  ([`ea00052`](https://github.com/sofianedjerbi/nexel-agent/commit/ea000523eee7262eaaed058ab36c757044f69a4d))

- Showcase readme with banner, shots and links; dev notes move to contributing
  ([`590fddb`](https://github.com/sofianedjerbi/nexel-agent/commit/590fddbd73e91bbe71308f0e8bd73829cc468581))

### Features

- **rl**: The 30B model talks and plans; a voice answers in about a second; rooms, gear and places
  ([`3bb879d`](https://github.com/sofianedjerbi/nexel-agent/commit/3bb879dca806cd4941abca146f3ee1b3168eebbf))


## v0.1.0 (2026-11-23)

The first tagged version: the AI players as they were when the history was rewritten to Conventional Commits.
