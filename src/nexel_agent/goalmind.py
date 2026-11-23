"""The goal side of the planner: the model's goal and its own done_when test (goals.py) checked after every action,
the model's "repeat" of its plan, and a plan that keeps failing said back as a fact. Mixed into Planner."""

import json
import time
from collections import Counter

from .goals import clean, missing
from .items import Item
from .plans import command_text

FAIL_WINDOW_S = 60.0


class GoalMind:
    def set_goal(self, goal, test):
        """A new goal and its own success test (goals.py), checked after every action."""
        if goal == self.goal and clean(test) == getattr(self, "goal_test", None):
            return                                   # (the same goal asked again: its clock and its count go on)
        if goal != self.goal and getattr(self, "goal_test", None):
            self.mem.add("goal", f"goal '{self.goal}' dropped for '{goal}'")
        self.goal, self.goal_test = goal, clean(test)
        self.goal_since, self.goal_t0 = dict(self.record.placed), time.time()

    def goal_line(self, obs) -> str:
        test = getattr(self, "goal_test", None)
        if not test:
            return f"Your goal '{self.goal}' has no done_when test." if self.goal else ""
        return (f"Your goal '{self.goal}' (done_when {json.dumps(test)}), for {int(time.time() - self.goal_t0)} s; "
                f"still missing: {'; '.join(missing(test, obs, self.record, self.goal_since)) or 'nothing'}.")

    def check_goal(self, obs):
        test = getattr(self, "goal_test", None)
        if test and not missing(test, obs, self.record, self.goal_since):
            secs = int(time.time() - self.goal_t0)
            self.tell(f"(goal '{self.goal}' done in {secs} s: {json.dumps(test)} holds)")
            self.mem.add("goal", f"goal '{self.goal}' done in {secs} s", None, (obs["self"]["x"], obs["self"]["y"]))
            self.decision_log.write(json.dumps({"kind": "goal", "t": time.time(), "goal": self.goal, "done": True,
                                                "test": test, "secs": secs}) + "\n")
            self.stats["goals_done"] += 1
            self.goal_test = None

    def replay(self) -> bool:
        """The plan said "repeat": with nothing else queued and its goal's test not yet holding, play it again."""
        rep = getattr(self, "repeat", None)
        test = getattr(self, "goal_test", None)
        if not rep or rep["left"] <= 0 or not test or not self.obs or not missing(test, self.obs, self.record, self.goal_since):
            return False
        rep["left"] -= 1
        self.queue.extend(Item(t, a, "model", f"repeat ({rep['left']} left)", None, self.epoch, rep["pid"]) for t, a in rep["cmds"])
        self.stats["repeats"] += 1
        return True

    def plan_failed(self, item, error):
        """A command of plan N failed: kept for the fact 'this plan failed k times in the last minute'; its repeat ends."""
        self.plan_fails = [f for f in getattr(self, "plan_fails", []) if time.time() - f[0] < FAIL_WINDOW_S]
        self.plan_fails.append((time.time(), tuple(getattr(self, "plans", {}).get(item.plan, ())), f"{command_text(item.tool, item.args)}: {error}"))
        if getattr(self, "repeat", None) and self.repeat["pid"] == item.plan:
            self.repeat = None

    def fail_facts(self) -> str:
        c = Counter(p for t, p, _ in getattr(self, "plan_fails", []) if p and time.time() - t < FAIL_WINDOW_S)
        last = {p: e for _, p, e in getattr(self, "plan_fails", [])}
        return " ".join(f"The plan '{', '.join(p)}' failed {n} times in the last minute (last: {last[p]})." for p, n in c.most_common(2) if n >= 2)
