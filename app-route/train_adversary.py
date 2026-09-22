#!/usr/bin/env python3
"""
Train / compare Route outer-curriculum adversaries (CPU-only for the adversary).

Modes:
  --mock          Simulate purple outcomes without Mininet / A2A (smoke + unit training).
  (default)       Drive live Mininet + fixed purple via evaluate_routing_queries.

Examples:
  python train_adversary.py --mock --curriculum sarsa --episodes 50 --horizon 8
  python train_adversary.py --mock --curriculum lm_mlp --episodes 40 --horizon 8
  python train_adversary.py --mock --curriculum verified_ac --episodes 40 --horizon 8
  python train_adversary.py --mock --compare
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

# Repo roots: app-route/ and src/
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_APP_DIR))

from netarena.adversary.arms import build_arms
from netarena.adversary.bandit import MyopicBandit
from netarena.adversary.lm_mlp_policy import LmMlpPolicy
from netarena.adversary.policy import TabularSarsa
from netarena.adversary.route_mdp import Outcome, RouteCurriculumEnv
from netarena.adversary.verified_ac import VerifiedActorCritic

NEURAL = ("lm_mlp", "verified_ac")
ALL_CURRICULA = ("random", "bandit", "sarsa", "lm_mlp", "verified_ac")


def _entropy(visits: list[int]) -> float:
    total = sum(visits)
    if total == 0:
        return 0.0
    h = 0.0
    for v in visits:
        if v == 0:
            continue
        p = v / total
        h -= p * math.log(p)
    return h


class MockPurple:
    """
    Fixed-target stub: fails on a designated hard-arm set (and rarely elsewhere).
    Used for CPU smoke training without Mininet.
    """

    def __init__(self, n_arms: int, hard_arms: set[int], *, hard_fail_p: float = 0.9, easy_fail_p: float = 0.2, seed: int = 0):
        if not hard_arms:
            raise ValueError("hard_arms must be non-empty")
        for a in hard_arms:
            if a < 0 or a >= n_arms:
                raise ValueError(f"hard arm {a} out of range [0, {n_arms})")
        self.n_arms = n_arms
        self.hard_arms = set(hard_arms)
        self.hard_fail_p = hard_fail_p
        self.easy_fail_p = easy_fail_p
        self._rng = random.Random(seed)

    def outcome(self, arm_id: int) -> Outcome:
        if arm_id < 0 or arm_id >= self.n_arms:
            raise ValueError(f"arm_id out of range: {arm_id}")
        p_fail = self.hard_fail_p if arm_id in self.hard_arms else self.easy_fail_p
        failed = self._rng.random() < p_fail
        unsafe = failed and self._rng.random() < (0.3 if arm_id in self.hard_arms else 0.05)
        return Outcome(correct=not failed, safe=not unsafe)


def _make_policy(curriculum: str, n_arms: int, args: argparse.Namespace, arms):
    if curriculum == "sarsa":
        return TabularSarsa(
            n_arms,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
        )
    if curriculum == "bandit":
        return MyopicBandit(
            n_arms,
            alpha=args.alpha,
            epsilon=args.epsilon,
            seed=args.seed,
        )
    if curriculum == "lm_mlp":
        return LmMlpPolicy(
            n_arms,
            lm_backend=args.lm_backend,
            lm_model_name=args.lm_model,
            embed_dim=args.embed_dim,
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            arms=arms,
        )
    if curriculum == "verified_ac":
        return VerifiedActorCritic(
            n_arms,
            lm_backend=args.lm_backend,
            lm_model_name=args.lm_model,
            embed_dim=args.embed_dim,
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            seed=args.seed,
            arms=arms,
        )
    if curriculum == "random":
        return None
    raise ValueError(f"Unknown curriculum {curriculum!r}")


def run_mock_episode(
    *,
    curriculum: str,
    env: RouteCurriculumEnv,
    policy,
    purple: MockPurple,
    rng: random.Random,
) -> dict:
    state = env.reset()
    rewards = []
    corrects = []
    early_actions = []
    late_actions = []
    horizon = env.horizon
    info = {"visits": [0] * env.n_arms, "fails": [0] * env.n_arms, "succs": [0] * env.n_arms}

    if curriculum == "random":
        for t in range(horizon):
            action = rng.randrange(env.n_arms)
            if t < max(1, horizon // 3):
                early_actions.append(action)
            else:
                late_actions.append(action)
            outcome = purple.outcome(action)
            _, reward, done, info = env.step(action, outcome)
            rewards.append(reward)
            corrects.append(1 if outcome.correct else 0)
            if done:
                break
    elif curriculum in NEURAL:
        action = policy.select_action(state, unvisited=env.unvisited_arms())
        for t in range(horizon):
            if t < max(1, horizon // 3):
                early_actions.append(action)
            else:
                late_actions.append(action)
            outcome = purple.outcome(action)
            next_state, reward, done, info = env.step(action, outcome)
            rewards.append(reward)
            corrects.append(1 if outcome.correct else 0)
            policy.update(state, action, reward, next_state, action, done=done)
            state = next_state
            if done:
                break
            action = policy.select_action(state, unvisited=env.unvisited_arms())
        policy.decay_epsilon()
    else:
        action = policy.select_action(state, unvisited=env.unvisited_arms())
        for t in range(horizon):
            if t < max(1, horizon // 3):
                early_actions.append(action)
            else:
                late_actions.append(action)
            outcome = purple.outcome(action)
            next_state, reward, done, info = env.step(action, outcome)
            rewards.append(reward)
            corrects.append(1 if outcome.correct else 0)
            if done:
                policy.update(state, action, reward, next_state, action, done=True)
                break
            next_action = policy.select_action(next_state, unvisited=env.unvisited_arms())
            policy.update(state, action, reward, next_state, next_action, done=False)
            state = next_state
            action = next_action
        policy.decay_epsilon()

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "purple_success_rate": sum(corrects) / len(corrects),
        "mean_adversary_failure_reward": sum(1 - c for c in corrects) / len(corrects),
        "entropy": _entropy(info["visits"]),
        "unique_arms": sum(1 for v in info["visits"] if v > 0),
        "early_entropy": _entropy([early_actions.count(a) for a in range(env.n_arms)]),
        "late_entropy": _entropy([late_actions.count(a) for a in range(env.n_arms)]),
        "visits": info["visits"],
        "fails": info["fails"],
        "succs": info["succs"],
    }


def train_mock(args: argparse.Namespace) -> Path:
    arms = build_arms()
    n_arms = len(arms)
    hard = set(args.hard_arms) if args.hard_arms else {0, 3, 7}
    purple = MockPurple(n_arms, hard, seed=args.seed)
    policy = _make_policy(args.curriculum, n_arms, args, arms)
    env = RouteCurriculumEnv(
        horizon=args.horizon,
        safety_weight=args.safety_weight,
        repeat_penalty=args.repeat_penalty,
        arms=arms,
    )
    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"mock_{args.curriculum}_metrics.jsonl"
    q_path = out_dir / f"mock_{args.curriculum}_q.json"

    episode_rows = []
    with metrics_path.open("w") as f:
        for ep in range(args.episodes):
            row = run_mock_episode(
                curriculum=args.curriculum,
                env=env,
                policy=policy,
                purple=purple,
                rng=rng,
            )
            row["episode"] = ep
            episode_rows.append(row)
            f.write(json.dumps(row) + "\n")
            if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
                print(
                    f"[{args.curriculum}] ep={ep+1}/{args.episodes} "
                    f"adv_fail={row['mean_adversary_failure_reward']:.3f} "
                    f"purple_ok={row['purple_success_rate']:.3f} "
                    f"H={row['entropy']:.3f} earlyH={row['early_entropy']:.3f} lateH={row['late_entropy']:.3f} "
                    f"unique={row['unique_arms']}"
                )

    if policy is not None:
        policy.save(q_path)

    summary = _summarize(episode_rows, args)
    summary_path = out_dir / f"mock_{args.curriculum}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    if policy is not None:
        print(f"Wrote {q_path}")
    return summary_path


def _summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    n = len(rows)
    mid = n // 2
    early = rows[: max(1, mid)]
    late = rows[mid:] if mid < n else rows

    def avg(key, subset):
        return sum(r[key] for r in subset) / len(subset)

    return {
        "curriculum": args.curriculum,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "hard_arms": args.hard_arms or [0, 3, 7],
        "overall_purple_success": avg("purple_success_rate", rows),
        "overall_adv_failure": avg("mean_adversary_failure_reward", rows),
        "early_episodes_adv_failure": avg("mean_adversary_failure_reward", early),
        "late_episodes_adv_failure": avg("mean_adversary_failure_reward", late),
        "early_episodes_arm_entropy": avg("early_entropy", early),
        "late_episodes_arm_entropy": avg("late_entropy", late),
        "mean_unique_arms": avg("unique_arms", rows),
    }


async def train_live(args: argparse.Namespace) -> None:
    from test_function import AppRouteConfig, evaluate_routing_queries
    from netarena.agent_client import AgentClientConfig, PromptType

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / f"live_{args.curriculum}_q.json"
    metrics_path = out_dir / f"live_{args.curriculum}_metrics.jsonl"

    agent_cfg = AgentClientConfig(
        base_url=args.purple_url,
        name="fixed-purple",
        prompt_type=PromptType(args.prompt_type),
    )

    rows = []
    with metrics_path.open("w") as f:
        for ep in range(args.episodes):
            result_dir = out_dir / f"ep_{ep:04d}"
            result_dir.mkdir(parents=True, exist_ok=True)
            cfg = AppRouteConfig(
                num_queries=args.horizon,
                output_dir=str(out_dir),
                max_iterations=args.max_iterations,
                curriculum=args.curriculum,
                curriculum_horizon=args.horizon,
                curriculum_q_path=str(q_path) if q_path.exists() or Path(str(q_path) + ".pt").exists() or Path(q_path).with_suffix(".pt").exists() else None,
                curriculum_save_q_path=str(q_path),
                curriculum_epsilon=args.epsilon,
                curriculum_alpha=args.alpha,
                curriculum_gamma=args.gamma,
                curriculum_safety_weight=args.safety_weight,
                curriculum_repeat_penalty=args.repeat_penalty,
                curriculum_seed=args.seed,
                curriculum_train=True,
                curriculum_lm_backend=args.lm_backend,
                curriculum_lm_model_name=args.lm_model,
                curriculum_lm_embed_dim=args.embed_dim,
                curriculum_lm_lr=args.lr,
                curriculum_entropy_coef=args.entropy_coef,
                curriculum_value_coef=args.value_coef,
                num_switches=args.num_switches,
                num_hosts_per_subnet=args.num_hosts_per_subnet,
                agent_client_configs=[agent_cfg],
            )
            ep_rewards = []
            ep_success = []
            last_curriculum = None
            async for ev in evaluate_routing_queries(cfg, result_dir=str(result_dir)):
                ep_success.append(1 if ev["success"] else 0)
                if "curriculum" in ev:
                    last_curriculum = ev["curriculum"]
                    ep_rewards.append(ev["curriculum"]["reward"])
            row = {
                "episode": ep,
                "purple_success_rate": sum(ep_success) / len(ep_success),
                "mean_reward": (sum(ep_rewards) / len(ep_rewards)) if ep_rewards else 0.0,
                "mean_adversary_failure_reward": 1.0 - (sum(ep_success) / len(ep_success)),
                "entropy": last_curriculum["entropy"] if last_curriculum else 0.0,
                "unique_arms": last_curriculum["unique_arms"] if last_curriculum else 0,
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(
                f"[{args.curriculum}] live ep={ep+1}/{args.episodes} "
                f"purple_ok={row['purple_success_rate']:.3f} H={row['entropy']:.3f}"
            )

    summary = {
        "curriculum": args.curriculum,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "overall_purple_success": sum(r["purple_success_rate"] for r in rows) / len(rows),
        "overall_adv_failure": sum(r["mean_adversary_failure_reward"] for r in rows) / len(rows),
    }
    (out_dir / f"live_{args.curriculum}_summary.json").write_text(json.dumps(summary, indent=2))


def compare_mock(args: argparse.Namespace) -> None:
    """Run all curricula mock trains and write a comparison JSON."""
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for curriculum in ALL_CURRICULA:
        sub = argparse.Namespace(**vars(args))
        sub.curriculum = curriculum
        sub.output_dir = str(base_out / curriculum)
        train_mock(sub)
        summary = json.loads((Path(sub.output_dir) / f"mock_{curriculum}_summary.json").read_text())
        summaries[curriculum] = summary

    compare_path = base_out / "compare_summary.json"
    compare_path.write_text(json.dumps(summaries, indent=2))
    print("\n=== Comparison (mock purple) ===")
    for k, s in summaries.items():
        print(
            f"{k:12s} purple_ok={s['overall_purple_success']:.3f} "
            f"adv_fail={s['overall_adv_failure']:.3f} "
            f"earlyH={s['early_episodes_arm_entropy']:.3f} lateH={s['late_episodes_arm_entropy']:.3f}"
        )
    print(f"Wrote {compare_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Route adversarial curriculum (CPU adversary).")
    p.add_argument("--curriculum", choices=list(ALL_CURRICULA), default="sarsa")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--safety-weight", type=float, default=0.25)
    p.add_argument("--repeat-penalty", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hard-arms", type=int, nargs="*", default=None, help="Mock purple hard arm ids")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--mock", action="store_true", help="Use MockPurple (no Mininet)")
    p.add_argument("--compare", action="store_true", help="Run all curricula mock comparison")
    p.add_argument("--purple-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--prompt-type", type=str, default="zeroshot_base")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--num-switches", type=int, default=2)
    p.add_argument("--num-hosts-per-subnet", type=int, default=1)
    p.add_argument("--lm-backend", choices=["hash", "hf"], default="hash")
    p.add_argument("--lm-model", type=str, default=None, help="HF model id when --lm-backend hf")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    args = p.parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(_APP_DIR / "output" / "adversary" / stamp)
    if args.lm_backend == "hf" and not args.lm_model:
        raise SystemExit("--lm-model is required when --lm-backend hf")
    return args


def main() -> None:
    args = parse_args()
    if args.compare:
        args.mock = True
        compare_mock(args)
        return
    if args.mock:
        train_mock(args)
        return
    import asyncio

    asyncio.run(train_live(args))


if __name__ == "__main__":
    main()
