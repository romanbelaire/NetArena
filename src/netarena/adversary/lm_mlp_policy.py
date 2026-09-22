"""LM body + MLP head policy over the discrete Route fault action set (bridge method)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from netarena.adversary.arms import RouteArm, build_arms
from netarena.adversary.lm_body import build_lm_body
from netarena.adversary.route_mdp import State
from netarena.adversary.state_text import state_to_text


class ActionHead(nn.Module):
    def __init__(self, embed_dim: int, n_arms: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_arms),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class LmMlpPolicy:
    """
    Bridge adversary: LM encodes belief text; MLP head scores the fixed arm set.
    Trained with REINFORCE on verified environment rewards (emulator / mock outcomes).
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
        seed: int | None = None,
        device: str = "cpu",
        arms: list[RouteArm] | None = None,
        trainable_lm: bool = False,
    ):
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if device != "cpu":
            raise ValueError("LmMlpPolicy must run on CPU (no GPU adversary training)")
        self.n_arms = n_arms
        self.gamma = gamma
        self.entropy_coef = entropy_coef
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
        self.head = ActionHead(body_dim, n_arms, hidden=hidden)
        self.body.to(self.device)
        self.head.to(self.device)

        params = list(self.head.parameters())
        if lm_backend == "hash" or trainable_lm:
            params += [p for p in self.body.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)

        self._traj_logps: list[torch.Tensor] = []
        self._traj_ents: list[torch.Tensor] = []
        self._traj_rewards: list[float] = []

    def _embed(self, state: State) -> torch.Tensor:
        text = state_to_text(state, self.arms)
        return self.body([text])

    def _dist(self, state: State) -> Categorical:
        logits = self.head(self._embed(state))
        return Categorical(logits=logits)

    def select_action(
        self,
        state: State,
        *,
        greedy: bool = False,
        unvisited: list[int] | None = None,
    ) -> int:
        del unvisited  # coverage handled by learned entropy / training signal
        dist = self._dist(state)
        if greedy:
            action = int(torch.argmax(dist.probs, dim=-1).item())
            return action
        action_t = dist.sample()
        self._traj_logps.append(dist.log_prob(action_t))
        self._traj_ents.append(dist.entropy())
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
        """Step API compatible with SARSA loop: accumulate verified reward; flush on done."""
        del state, action, next_state, next_action
        self.observe_reward(reward)
        if done:
            self.flush_episode()

    def flush_episode(self) -> dict:
        if len(self._traj_rewards) == 0:
            raise RuntimeError("flush_episode called with empty trajectory")
        if len(self._traj_logps) != len(self._traj_rewards):
            raise RuntimeError(
                f"logp/reward length mismatch: {len(self._traj_logps)} vs {len(self._traj_rewards)}"
            )

        returns = []
        g = 0.0
        for r in reversed(self._traj_rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        logps = torch.stack(self._traj_logps)
        ents = torch.stack(self._traj_ents)
        # REINFORCE on verified returns
        loss = -(logps * ret_t).mean() - self.entropy_coef * ents.mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        stats = {
            "policy_loss": float(loss.item()),
            "mean_return": float(ret_t.mean().item()),
            "entropy": float(ents.mean().item()),
        }
        self._traj_logps.clear()
        self._traj_ents.clear()
        self._traj_rewards.clear()
        return stats

    def decay_epsilon(self) -> None:
        return

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = path.with_suffix(".pt")
        meta = {
            "kind": "lm_mlp",
            "n_arms": self.n_arms,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "lm_backend": self.lm_backend,
            "lm_model_name": self.lm_model_name,
            "trainable_lm": self.trainable_lm,
            "embed_dim": int(getattr(self.body, "embed_dim")),
        }
        torch.save(
            {
                "head": self.head.state_dict(),
                "body": self.body.state_dict(),
                "meta": meta,
            },
            ckpt,
        )
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "LmMlpPolicy":
        path = Path(path)
        ckpt_path = path.with_suffix(".pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Expected torch checkpoint at {ckpt_path}")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = blob["meta"]
        agent = cls(
            meta["n_arms"],
            lm_backend=meta["lm_backend"],
            lm_model_name=meta["lm_model_name"],
            embed_dim=meta["embed_dim"],
            gamma=meta["gamma"],
            entropy_coef=meta["entropy_coef"],
            seed=seed,
            trainable_lm=meta["trainable_lm"],
        )
        agent.head.load_state_dict(blob["head"])
        agent.body.load_state_dict(blob["body"])
        return agent
