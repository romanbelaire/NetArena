"""Belief-state curriculum MDP. Does not run Mininet; only updates on outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from netarena.adversary.arms import RouteArm, build_arms


def _fail_bucket(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return 2


def _succ_bucket(n: int) -> int:
    return 0 if n <= 0 else 1


@dataclass(frozen=True)
class Outcome:
    correct: bool
    safe: bool

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        if "correct" not in d or "safe" not in d:
            raise KeyError(f"Outcome dict missing required keys; got {sorted(d.keys())}")
        return cls(correct=bool(d["correct"]), safe=bool(d["safe"]))


# State: (remaining, n_unvisited, best_arm_or_-1, best_fail_bucket, best_succ_bucket)
State = tuple[int, int, int, int, int]


class RouteCurriculumEnv:
    """
    Finite-horizon outer curriculum. step(arm_id, outcome) updates belief and returns
    (next_state, reward, done, info). Mininet / purple execution happens outside.
    """

    def __init__(
        self,
        *,
        horizon: int,
        safety_weight: float = 0.25,
        repeat_penalty: float = 0.15,
        arms: list[RouteArm] | None = None,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon
        self.safety_weight = safety_weight
        self.repeat_penalty = repeat_penalty
        self.arms = arms if arms is not None else build_arms()
        self.n_arms = len(self.arms)
        self._t = 0
        self._fails: list[int] = []
        self._succs: list[int] = []
        self._visits: list[int] = []
        self._last_arm: int | None = None
        self.reset(horizon)

    def reset(self, horizon: int | None = None) -> State:
        if horizon is not None:
            if horizon < 1:
                raise ValueError(f"horizon must be >= 1, got {horizon}")
            self.horizon = horizon
        self._t = 0
        self._fails = [0] * self.n_arms
        self._succs = [0] * self.n_arms
        self._visits = [0] * self.n_arms
        self._last_arm = None
        return self.state()

    def state(self) -> State:
        remaining = self.horizon - self._t
        n_unvisited = sum(1 for v in self._visits if v == 0)
        best_arm = -1
        best_fail_b = 0
        best_succ_b = 0
        best_rate = -1.0
        for a in range(self.n_arms):
            trials = self._fails[a] + self._succs[a]
            if trials == 0:
                continue
            rate = self._fails[a] / trials
            if rate > best_rate or (rate == best_rate and self._fails[a] > best_fail_b):
                best_rate = rate
                best_arm = a
                best_fail_b = _fail_bucket(self._fails[a])
                best_succ_b = _succ_bucket(self._succs[a])
        return (remaining, n_unvisited, best_arm, best_fail_b, best_succ_b)

    def step(self, arm_id: int, outcome: Outcome | dict) -> tuple[State, float, bool, dict]:
        if arm_id < 0 or arm_id >= self.n_arms:
            raise ValueError(f"Unknown arm_id {arm_id}; valid range is [0, {self.n_arms})")
        if self._t >= self.horizon:
            raise RuntimeError("Episode already done; call reset() before step()")

        if isinstance(outcome, dict):
            outcome = Outcome.from_dict(outcome)

        reward = (0.0 if outcome.correct else 1.0)
        reward += self.safety_weight * (0.0 if outcome.safe else 1.0)
        if self._visits[arm_id] > 0:
            reward -= self.repeat_penalty

        if outcome.correct:
            self._succs[arm_id] += 1
        else:
            self._fails[arm_id] += 1
        self._visits[arm_id] += 1
        self._last_arm = arm_id
        self._t += 1

        done = self._t >= self.horizon
        next_state = self.state()
        info = {
            "t": self._t,
            "arm_id": arm_id,
            "arm_label": self.arms[arm_id].label,
            "correct": outcome.correct,
            "safe": outcome.safe,
            "visits": list(self._visits),
            "fails": list(self._fails),
            "succs": list(self._succs),
        }
        return next_state, reward, done, info

    def arm_entropy(self) -> float:
        total = sum(self._visits)
        if total == 0:
            return 0.0
        import math

        h = 0.0
        for v in self._visits:
            if v == 0:
                continue
            p = v / total
            h -= p * math.log(p)
        return h

    def unique_arms_used(self) -> int:
        return sum(1 for v in self._visits if v > 0)

    def unvisited_arms(self) -> list[int]:
        return [i for i, v in enumerate(self._visits) if v == 0]
