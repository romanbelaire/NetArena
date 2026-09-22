"""Serialize Route curriculum belief state as text for LM bodies."""

from __future__ import annotations

from netarena.adversary.arms import RouteArm
from netarena.adversary.route_mdp import State


def state_to_text(state: State, arms: list[RouteArm] | None = None) -> str:
    remaining, n_unvisited, best_arm, best_fail_b, best_succ_b = state
    best_label = "none"
    if best_arm >= 0 and arms is not None:
        if best_arm >= len(arms):
            raise ValueError(f"best_arm {best_arm} out of range for {len(arms)} arms")
        best_label = arms[best_arm].label
    elif best_arm >= 0:
        best_label = str(best_arm)
    return (
        "Route curriculum belief state.\n"
        f"Remaining budget: {remaining}.\n"
        f"Unvisited arms: {n_unvisited}.\n"
        f"Best known hard arm: {best_label} "
        f"(fail_bucket={best_fail_b}, succ_bucket={best_succ_b}).\n"
        "Choose the next fault injection arm to maximize purple failure "
        "while probing unknown arms early and exploiting later."
    )
