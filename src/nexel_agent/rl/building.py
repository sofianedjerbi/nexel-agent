"""The mind's side of its home and its building (#30): the home the body chose kept across sessions, the build the
planner set steered to its end (the body lays it: rl/build.py), how the home stands told and graded. A mixin of
rl/mind.py's Mind (its state: home, social, building, build_end, home_read, tried, ended)."""

from . import home
from .facts import counted

BUILD = "build your home"     # (the build goal: no goal kind the body knows, the mind's own - steer)
BUILT = 5                    # a home built: the judge's v1 at 5 of its 6 checks or more (rl/judge.py CHECKS)
LACKING = {"closed": "walls all round", "door": "a door", "roofed": "a roof", "lit": "a light", "bed": "a bed",
           "room_enough": "room inside"}


class Building:
    def keep_home(self):
        """The home the body chose (rl/build.py social.home), kept: at the start, the one it had last session is handed
        back to a body that has none; after that it is the body's, kept as it is (dropped when the body gives it up)."""
        s = self.social
        if s is None:
            return
        if not self.home_back:
            self.home_back = True
            if s.home is None and self.home:
                s.home = self.home
        self.home = tuple(s.home) if s.home else None

    def steer(self, i: int, mask, hurt: float) -> int:
        """The body's action, or build while the mind is building its home (the planner's call: goal 'build'): the
        learned body has not learned to finish a hut (r51-r65), the build verb does it (rl/build.py). Hurt, the body
        acts as it learned (a fight, a flight); build with nothing to lay ends the building, told in the next facts."""
        from .actions import ACTIONS
        b = next(k for k, a in enumerate(ACTIONS) if a[0] == "build")
        if not self.building or hurt > 0:
            return i
        if mask is not None and not mask[b]:
            self.stop_building("nothing more it can lay or make here now")
            return i
        return b

    def stop_building(self, why: str):
        """The building ends: told and tallied in the next facts, with how the home stands then (built)."""
        self.building, self.build_end, self.ended = False, why, f"{BUILD}: {why}"

    def stands(self) -> tuple:
        """How its home stands, from the home's reading (rl/home.py reading: the judge's v1 on the observation, as the
        home facts line says it): ("it stands at 4/6, lacking a roof", built: 5 of 6 or more)."""
        r = self.home_read
        if not self.home:
            return "you have no home yet", False
        if not r or r.get("far"):
            return "your home is too far to see how it stands", False
        if "pieces" in r:                                    # (a drawn build: whole once every piece is done)
            n, of = r["pieces"]
            return (f"{n} of {of} pieces done" if n < of else "it stands whole"), n == of
        v = r["verdict"]
        lack = [LACKING.get(k, k.replace("_", " ")) for k in v["failed"]]
        return f"it stands at {v['grade']}/{v['of']}" + (f", lacking {', '.join(lack)}" if lack else ", whole"), v["grade"] >= BUILT

    def built(self):
        """The building that ended since the last facts, graded: done when the home stands at 5 of 6 or more."""
        if not self.build_end:
            return
        how, done = self.stands()
        self.ended = f"{BUILD}: {how}; {self.build_end}"
        tally = self.tried.setdefault(BUILD, [0, 0])
        tally[0] += 1
        tally[1] += done
        self.build_end = ""

    def home_needs(self, obs: dict) -> str:
        """What its home still takes, for the aim "home": its lacks beyond the pack, or that it has not chosen one."""
        if not self.home:
            return "you have no home yet: find a flat spot and build"
        lack = home.lacks(obs, self.home, self.social)
        return f"it lacks {counted(lack)}" if lack else "you have all it takes"
