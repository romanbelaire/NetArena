"""Myopic contextual bandit (gamma=0 SARSA / sample-mean on arm features)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from netarena.adversary.route_mdp import State


def _state_key(state: State) -> str:
    return ",".join(str(x) for x in state)


class MyopicBandit:
    """
    Same state encoding as TabularSarsa, but updates with gamma=0 (immediate reward only).
    Exploration via epsilon-greedy.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        alpha: float = 0.1,
        epsilon: float = 0.2,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms = n_arms
        self.alpha = alpha
        self.gamma = 0.0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(seed)
        self.q: dict[str, list[float]] = {}

    def _ensure(self, state: State) -> list[float]:
        key = _state_key(state)
        if key not in self.q:
            self.q[key] = [0.0] * self.n_arms
        return self.q[key]

    def select_action(
        self,
        state: State,
        *,
        greedy: bool = False,
        unvisited: list[int] | None = None,
    ) -> int:
        remaining, n_unvisited, best_arm, *_ = state
        if (
            not greedy
            and unvisited
            and n_unvisited > 0
            and (best_arm < 0 or remaining < n_unvisited)
            and self._rng.random() < max(0.5, self.epsilon)
        ):
            return self._rng.choice(unvisited)
        values = self._ensure(state)
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_arms)
        max_q = max(values)
        best = [i for i, v in enumerate(values) if v == max_q]
        return self._rng.choice(best)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        *,
        done: bool,
    ) -> None:
        del next_state, next_action, done  # myopic: ignore bootstrap
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"action out of range: {action}")
        q_sa = self._ensure(state)
        q_sa[action] += self.alpha * (reward - q_sa[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "n_arms": self.n_arms,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q": self.q,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "MyopicBandit":
        payload: dict[str, Any] = json.loads(Path(path).read_text())
        agent = cls(
            payload["n_arms"],
            alpha=payload["alpha"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            seed=seed,
        )
        agent.q = {k: list(v) for k, v in payload["q"].items()}
        return agent
