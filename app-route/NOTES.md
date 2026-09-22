# Route notes

## 2026-03-22 — LM+MLP bridge and verified actor-critic — local mock — done

Added two neural adversary instantiations on top of tabular SARSA/bandit:

1. `lm_mlp` — LM body (default CPU `hash` bag-of-ngrams bridge; optional HuggingFace encoder via `--lm-backend hf`) plus MLP head over the fixed 15-arm Route fault set. Trained with REINFORCE on verified emulator/mock rewards.
2. `verified_ac` — shared LM body, MLP policy head, MLP value head. Critic targets are verified Monte Carlo returns from environment outcomes only (no LLM-as-judge). Actor uses advantages G_t - V(s_t).

Smoke (mock purple, hash LM body, CPU, 30 episodes x horizon 8): both curricula run end-to-end and checkpoint `.pt` weights. Live Mininet+purple still requires a privileged host.

## 2026-03-21 — adversarial outer curriculum (MDP) — local mock — done

We added an online probe-then-exploit curriculum inside the Route green loop so the adversary (task chooser) is an RL policy and the purple agent is a fixed target. The outer step chooses a discrete fault arm, runs the existing Mininet diagnose-fix episode, observes success/safety, updates a belief state, and picks the next arm to maximize purple failure over a finite horizon.

Implementation lives in `src/netarena/adversary/` (arms, belief MDP, tabular SARSA, myopic bandit) and is wired through `curriculum` on `AppRouteConfig` in `test_function.py`. Training entrypoint: `python train_adversary.py --mock --compare`.

Smoke comparison (CPU mock purple that fails on hard arms `{0,3,7}`; 80 episodes x horizon 10; seed 1; no Mininet on this login node):

| Curriculum | Purple success | Adv failure rate | Late-ep adv failure |
|---|---:|---:|---:|
| random | 0.685 | 0.315 | 0.295 |
| bandit | 0.610 | 0.390 | 0.428 |
| sarsa | 0.629 | 0.371 | 0.385 |

Relative to random sampling, both adaptive curricula raise the adversary's failure rate (lower purple success), matching the hypothesis that a fixed purple overfits a static/random fault distribution. Bandit shows the clearest learning signal (late-episode adv failure above early). Full Mininet + live purple was not run here because Mininet is unavailable on the login node; use `train_adversary.py` without `--mock` on a privileged Route container when ready. Artifacts: `output/adversary/smoke_final/compare_summary.json`.
