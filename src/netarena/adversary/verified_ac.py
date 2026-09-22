"""
Verified-reward actor-critic with a verified value function.

Rewards and return targets come only from emulator/mock outcomes (not LLM judges).
The critic V(s) is fit to those verified Monte Carlo returns; the LM+MLP actor
optimizes advantages (G_t - V(s_t)).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from netarena.adversary.arms import RouteArm, build_arms
from netarena.adversary.lm_body import HashLmBody, build_lm_body
from netarena.adversary.lm_mlp_policy import ActionHead
from netarena.adversary.route_mdp import State
from netarena.adversary.state_text import state_to_text


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


class VerifiedActorCritic:
    """
    Shared LM body, MLP policy head over discrete arms, MLP value head.
    Critic targets = verified discounted returns from environment outcomes.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        lm_backend: str = "hash",
        lm_model_name: str | None = None,
        embed_dim: int = 128,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: int | None = None,
        device: str = "cpu",
        arms: list[RouteArm] | None = None,
        trainable_lm: bool = False,
    ):
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if device != "cpu":
            raise ValueError("VerifiedActorCritic must run on CPU (no GPU adversary training)")
        self.n_arms = n_arms
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)
        self.arms = arms if arms is not None else build_arms()
        if len(self.arms) != n_arms:
            raise ValueError(f"arms length {len(self.arms)} != n_arms {n_arms}")
        self.lm_backend = lm_backend
        self.lm_model_name = lm_model_name
        self.trainable_lm = trainable_lm

        if seed is not None:
            torch.manual_seed(seed)

        self.body = build_lm_body(
            lm_backend,
            embed_dim=embed_dim,
            model_name=lm_model_name,
            seed=0 if seed is None else seed,
            device=device,
            trainable_lm=trainable_lm,
        )
        body_dim = int(getattr(self.body, "embed_dim"))
        self.policy_head = ActionHead(body_dim, n_arms, hidden=hidden)
        self.value_head = ValueHead(body_dim, hidden=hidden)
        self.body.to(self.device)
        self.policy_head.to(self.device)
        self.value_head.to(self.device)

        params = list(self.policy_head.parameters()) + list(self.value_head.parameters())
        if lm_backend == "hash" or trainable_lm:
            params += [p for p in self.body.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)

        self._traj_logps: list[torch.Tensor] = []
        self._traj_values: list[torch.Tensor] = []
        self._traj_ents: list[torch.Tensor] = []
        self._traj_rewards: list[float] = []

    def _embed(self, state: State) -> torch.Tensor:
        text = state_to_text(state, self.arms)
        return self.body([text])

    def select_action(
        self,
        state: State,
        *,
        greedy: bool = False,
        unvisited: list[int] | None = None,
    ) -> int:
        del unvisited
        emb = self._embed(state)
        logits = self.policy_head(emb)
        value = self.value_head(emb)
        dist = Categorical(logits=logits)
        if greedy:
            return int(torch.argmax(dist.probs, dim=-1).item())
        action_t = dist.sample()
        self._traj_logps.append(dist.log_prob(action_t))
        self._traj_ents.append(dist.entropy())
        self._traj_values.append(value.squeeze(0))
        return int(action_t.item())

    def observe_reward(self, reward: float) -> None:
        self._traj_rewards.append(float(reward))

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
        del state, action, next_state, next_action
        self.observe_reward(reward)
        if done:
            self.flush_episode()

    def flush_episode(self) -> dict:
        if len(self._traj_rewards) == 0:
            raise RuntimeError("flush_episode called with empty trajectory")
        n = len(self._traj_rewards)
        if not (
            len(self._traj_logps) == n
            and len(self._traj_values) == n
            and len(self._traj_ents) == n
        ):
            raise RuntimeError(
                f"trajectory length mismatch: rewards={n} "
                f"logps={len(self._traj_logps)} values={len(self._traj_values)} ents={len(self._traj_ents)}"
            )

        # Verified Monte Carlo returns from environment rewards only.
        returns = []
        g = 0.0
        for r in reversed(self._traj_rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        logps = torch.stack(self._traj_logps)
        values = torch.stack(self._traj_values)
        ents = torch.stack(self._traj_ents)

        advantages = ret_t - values.detach()
        policy_loss = -(logps * advantages).mean() - self.entropy_coef * ents.mean()
        # Verified value function: regress V(s) onto verified returns.
        value_loss = ((values - ret_t) ** 2).mean()
        loss = policy_loss + self.value_coef * value_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_return": float(ret_t.mean().item()),
            "mean_value": float(values.mean().item()),
            "entropy": float(ents.mean().item()),
        }
        self._traj_logps.clear()
        self._traj_values.clear()
        self._traj_ents.clear()
        self._traj_rewards.clear()
        return stats

    def decay_epsilon(self) -> None:
        return

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = path.with_suffix(".pt") if path.suffix != ".pt" else path
        meta_path = ckpt.with_suffix(".json")
        torch.save(
            {
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "body": self.body.state_dict(),
                "meta": {
                    "kind": "verified_ac",
                    "n_arms": self.n_arms,
                    "gamma": self.gamma,
                    "entropy_coef": self.entropy_coef,
                    "value_coef": self.value_coef,
                    "lm_backend": self.lm_backend,
                    "lm_model_name": self.lm_model_name,
                    "trainable_lm": self.trainable_lm,
                    "embed_dim": int(getattr(self.body, "embed_dim")),
                },
            },
            ckpt,
        )
        import json

        meta_path.write_text(json.dumps({"checkpoint": str(ckpt.name), "kind": "verified_ac"}, indent=2))

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "VerifiedActorCritic":
        path = Path(path)
        ckpt = path.with_suffix(".pt") if path.suffix != ".pt" else path
        if not ckpt.exists():
            # allow passing the .json sidecar
            alt = path.with_suffix(".pt")
            if alt.exists():
                ckpt = alt
            else:
                raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = blob["meta"]
        agent = cls(
            meta["n_arms"],
            lm_backend=meta["lm_backend"],
            lm_model_name=meta["lm_model_name"],
            embed_dim=meta["embed_dim"],
            gamma=meta["gamma"],
            entropy_coef=meta["entropy_coef"],
            value_coef=meta["value_coef"],
            seed=seed,
            trainable_lm=meta["trainable_lm"],
        )
        agent.policy_head.load_state_dict(blob["policy_head"])
        agent.value_head.load_state_dict(blob["value_head"])
        agent.body.load_state_dict(blob["body"])
        return agent
