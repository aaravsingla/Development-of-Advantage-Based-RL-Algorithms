# Advantage Estimation, Gradient Variance & Safe Reward Balancing — Project Plan

## Narrative

One story ties every resume claim together:

> **How the advantage estimator and the reward design control policy-gradient
> variance and update stability — and why that matters once safety penalties
> enter the reward.**

"Safe Reward Balancing" (our definition): the trade-off in the safety-penalty
weight. Too large → large, heavy-tailed advantages → unstable updates; too
small → unsafe behaviour. Dense CBF-based shaping is the proposed fix.

## Resume claims → evidence

| Claim | Evidence produced by | Phase |
|---|---|---|
| GAE implemented from scratch | `rl/gae.py` + brute-force unit tests | 1 |
| GAE reduces gradient variance without significant bias | Direct gradient bias²/variance/MSE on frozen policies | 2 |
| λ-ablation quantifies bias–variance trade-off | 8-λ sweep, learning curves + gradient stats | 2 |
| λ stabilises learning in noisy reward settings | Reward-noise sweep; theory σ²/(1−(γλ)²) vs measured | 2 |
| λ and value overestimation | Critic V(s) vs realised discounted return, per λ | 2 |
| On-policy (PPO) vs off-policy (DDPG) value-estimation variance | DDPG/TD3 from scratch, Q-bias & variance plots | 3 |
| Advantage magnitude ↔ update instability | Per-update logs, Spearman + interventions | 4 |
| CBF reward shaping, convergence speed, collision rate | Safe-nav env, 5 baselines, 10 seeds | 5 |

**Rule:** a claim only goes on the resume if a figure in `results/` backs it,
with the number taken from that figure (IQM + 95% bootstrap CI).

## Stack

Python 3.12, PyTorch (CPU — faster than MPS for small MLPs), Gymnasium
(Box2D + MuJoCo), NumPy, pandas, matplotlib, scipy, pytest, TensorBoard.
Seeds run in parallel via multiprocessing, `torch.set_num_threads(1)` per worker.

## Repo layout

```
ppo_ddpg_GAE/
  rl/            gae.py, ppo.py, networks.py, buffers.py, utils.py, (ddpg.py)
  envs/          wrappers.py (reward noise), (safe_nav.py, cbf.py)
  instrumentation/ grad_probe.py (gradient bias/variance), value_probe.py
  experiments/   configs/*.yaml, train_ppo.py, sweep.py, probe_gradients.py
  analysis/      plotting + statistics
  tests/         test_gae.py, test_ppo_smoke.py
  results/       runs/, figures/, tables/
```

---

## Phase 0 — Setup (≈1 h)
- venv, dependencies, repo skeleton, seeded parallel sweep launcher, logging (CSV + TensorBoard).

## Phase 1 — PPO + GAE from scratch, tested (≈4–6 h)
- `gae.py`: standalone, vectorised over envs; separate `terminated` vs `truncated`
  handling; bootstraps the value at rollout cut-off and at time-limit truncation.
- Unit tests vs brute-force Σ(γλ)^l δ_{t+l}; edge cases λ=0 (TD error), λ=1 (MC − V),
  termination, truncation, multi-env.
- PPO: learnable state-independent log-std, minibatches, advantages computed once
  per rollout, advantage normalisation (toggle), value clipping (toggle),
  observation normalisation, grad-norm clipping, logs approx-KL / clip-fraction /
  explained variance / raw advantage stats.
- Sanity: reach reference-range returns on Hopper-v5 and LunarLanderContinuous-v3.

## Phase 2 — GAE λ ablation with direct bias/variance measurement (≈5–7 h + compute)
1. **Learning curves**: λ ∈ {0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0}, 2 envs, 5 seeds; IQM + bootstrap CI.
2. **Gradient bias/variance probe** (the core result):
   - Freeze policy+critic checkpoints (early / mid / late) from a λ=0.95 run, so all
     λ are compared on the *same* policy — only the estimator changes.
   - Per λ: N independent batches → policy gradient ĝ_i = ∇ mean(log π · Â).
   - Reference g* = mean λ=1 gradient over a very large sample (unbiased; baseline adds no bias).
   - Report Var = E‖ĝ − E ĝ‖², Bias² = ‖E ĝ − g*‖², MSE, cosine(ĝ, g*).
   - Expect: variance ↓ and bias² ↑ as λ ↓ → U-shaped MSE; bias larger with an early (poor) critic.
3. **Noisy reward**: Gaussian reward noise σ ∈ {0, 0.1, 0.5, 1.0} × λ ∈ {0.5, 0.9, 0.95, 1.0};
   train on noisy reward, evaluate on clean. Theory: reward-noise contribution to Var(Â_t)
   is σ²·Σ(γλ)^{2l} = σ²/(1−(γλ)²) — verify empirically.
4. **Value overestimation (honest)**: critic V(s) vs realised discounted return on
   held-out rollouts, per λ. Report what the data shows; reword resume if needed.

## Phase 3 — PPO vs DDPG value estimation (≈5–7 h + compute)
- DDPG from scratch + TD3 flag (clipped double-Q, target smoothing, delayed actor).
- Q(s, μ(s)) vs realised discounted return along evaluation rollouts (Fujimoto-style).
- Compare bias and across-seed / across-state variance of PPO V vs DDPG Q vs TD3 Q.

## Phase 4 — Advantage magnitude ↔ update instability (≈3–4 h)
- Per update: raw |A| mean/std/p99/kurtosis vs approx-KL, clip-frac, grad-norm, post-update return drop.
- Spearman ρ with bootstrap CI; binned plots.
- Causal interventions: adv-norm on/off; reward scale ×{0.1, 1, 10}; safety-penalty weight sweep.

## Phase 5 — Safe navigation with CBF reward shaping (≈7–10 h + compute)
- Vectorised NumPy 2D point-robot env, random circular obstacles, lidar-like obs.
- h(x) = ‖p − p_obs‖² − (r + margin)²; shaping −k·max(0, (1−α)h(x_t) − h(x_{t+1})).
- Baselines: task-only, sparse collision penalty, distance penalty, **CBF shaping**,
  PPO-Lagrangian; reference: CBF-QP safety filter at execution.
- Equal tuning budget per method, 10 seeds.
- Metrics (fixed before running): success rate, collision rate, env-steps to ≥80% success.
- Optional: potential-based CBF variant (policy invariance, Ng et al. 1999).

## Phase 6 — Analysis & write-up (≈4–6 h)
- One script regenerates every figure/table. README contains only figure-backed claims.
- Final resume bullets with measured numbers.

## Phase 7 — Interview prep (≈4–6 h)
- Derive GAE from n-step returns; why PPO surrogate gradient at θ_old = PG;
  CBF forward invariance; why DDPG overestimates; defend each design choice.

## Time budget (M5, Claude Code)

| Phase | Hands-on | Compute |
|---|---|---|
| 0–1 | 5–7 h | <1 h |
| 2 | 5–7 h | 3–4 h |
| 3 | 5–7 h | 3–4 h |
| 4 | 3–4 h | reuses runs |
| 5 | 7–10 h | 2–3 h |
| 6–7 | 8–12 h | — |
| **Total** | **35–45 h** | **10–15 h (×2 for reruns)** |

## Status
- [x] Phase 0: venv (Python 3.12, torch 2.14, gymnasium 1.4, MuJoCo 3.15, Box2D), repo skeleton, resumable parallel sweep launcher
- [x] Phase 1: GAE + PPO from scratch; 35 tests pass; sanity: Hopper ~2000 and LunarLander ~130 by 250k steps
- [x] Phase 2: 140 training runs + 6 gradient probes + value probe; results in README
- [x] Phase 3: DDPG/TD3 from scratch, 30 runs (DDPG, TD3, twin-Q ablation); results in README
- [x] Phase 4: correlations over ~39k updates + 2 intervention sweeps (50 runs); results in README
- [x] Phase 5: SafeNav env + CBF tools, 98 runs (tuning + 10-seed final) + filtered eval; results in README
- [x] Phase 6: README restructured (key findings, limitations, full reproduce), make_figures.sh
- [ ] Phase 7
