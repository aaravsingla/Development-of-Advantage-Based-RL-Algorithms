# Advantage Estimation, Value Bias & Safe Reward Design in Policy-Gradient RL

From-scratch PPO, GAE, DDPG and TD3 (PyTorch), used to **measure directly** the quantities usually only inferred from learning curves:
- gradient bias and variance
- critic over/under-estimation
- how update size depends on the advantages
- whether CBF reward shaping helps safe navigation

**Scale:** about 320 training runs, 5–10 seeds per configuration. Results are reported as IQM with 95% bootstrap CIs.
**Tests:** 43 unit and integration tests.
**Rule:** every number below is backed by a figure in `results/figures/` and a table in `results/tables/`.

## Key findings

1. **GAE's bias–variance trade-off, measured in the gradient itself.**
   - On frozen policies, λ=0.95 has 4.3–7.5× lower gradient MSE than Monte Carlo (λ=1).
   - Variance falls monotonically as λ drops, and bias² rises (`p2_gradient_bias_variance.png`).
2. **Lowest gradient error ≠ best learning.**
   - λ=0 has the lowest gradient MSE but learns worst, because during training a low-λ critic bootstraps its own errors.
   - λ=0.95 beats Monte Carlo by 39% on Hopper and ties it on LunarLander.
3. **Reward-noise law verified.**
   - The noise variance in Â matches σ²/(1−(γλ)²) for every (λ, σ) tested.
   - Under mild reward noise, λ=0.95 keeps 93% of its clean return vs 69% for Monte Carlo.
4. **Off-policy overestimation, and what fixes it.**
   - DDPG overestimates values by 47–79%.
   - Clipped double-Q alone removes this (−3 to −5%) and doubles Hopper return.
   - PPO's critic is unbiased (+0.1%).
5. **Advantage magnitude causes instability only without normalisation.**
   - With normalisation off, the KL-spike rate rises from 0.5% to 49% as the reward scale grows 100×.
   - Normalisation removes the instability.
   - Safety penalties make advantages 4× heavier-tailed. The real failure is a safety–return *cliff* in the penalty weight, not unstable updates.
6. **CBF reward shaping did not speed up safe learning, but a CBF safety filter at deployment removed all collisions.**
   - The comparison used a pre-registered metric, equal tuning budget and 10 seeds.
   - CBF shaping vs sparse penalty: +0.6% faster, 95% CI [−37%, +22%].
   - The CBF-QP filter at deployment gave 0 collisions for every policy, with +2–4 points success.

## Limitations

- **Small tasks.** Two Gymnasium tasks (Hopper, LunarLander) and one custom 2D environment, 1M steps (500k for SafeNav). Results may not transfer to high-dimensional or image-based control.
- **Partial bias resolution.** The gradient probe resolves bias² only to the precision of 100 batches, and on LunarLander it is below resolution after early training.
- **Simplified Lagrangian.** PPO-Lagrangian uses one combined critic (penalty form), not a separate cost critic.
- **Untested critic hypothesis.** The critic/shared-gradient-clipping explanation in Phase 4 has not been confirmed by a separate-optimiser ablation.
- **SafeNav ends episodes on collision.** This gives an implicit penalty, so the null result for shaping may not hold in settings where collisions are not terminal.

## Results (Phase 2)

All results: PPO, 1M steps, 5 seeds per config, IQM with 95% bootstrap CI.

**1. GAE trades bias for variance in the gradient itself** (`p2_gradient_bias_variance.png`; frozen λ=0.95 policies, 3 seeds × 3 checkpoints, 100 batches each)
- Gradient variance falls steadily as λ decreases, relative to Monte Carlo (λ=1):
  - λ=0.95: 0.19–0.22× on Hopper, 0.13–0.17× on LunarLander
  - λ=0: ~0.02–0.05×
- On Hopper, bias² rises as λ decreases (≈0.015 at λ=0.95, ≈0.10 at λ=0), so MSE is U-shaped. The minimum is at λ≈0.5–0.8.
- At λ=0.95, gradient MSE is 4.3–4.8× lower than Monte Carlo on Hopper, and 5.7–7.5× lower on LunarLander.
- On LunarLander, bias² is below what 100 batches can resolve after early training.
- Late in training, ‖true gradient‖² is only ~0–14% of the variance of a single Monte Carlo batch (`snr_mc`).

**2. Lower gradient MSE ≠ better learning** (`p2_learning_curves.png`, `p2_lambda_final_return.png`)

| Final return | λ=0 | λ=0.8 | λ=0.95 | λ=1 |
|---|---|---|---|---|
| Hopper | 592 | 2063 | **2719** [2399, 3256] | 1959 [1472, 2440] |
| LunarLander | −189 | −10 | 190 [161, 199] | **216** [195, 232] |

- On Hopper, λ=0.95 beats Monte Carlo by about 39%.
- On LunarLander, λ ≥ 0.95 are statistically tied, and λ ≤ 0.8 fails, even though the probe says low λ has the *lowest* gradient MSE.
- The reason: the probe judges every λ with the same good critic. In training, a low-λ agent fits its critic to its own bootstrapped TD targets, so bootstrapping bias compounds over training.

**3. Reward noise** (`p2_noise_theory.png`, `p2_noise_learning.png`)
- The measured advantage-noise variance matches σ²/(1−(γλ)²) for every (λ, σ) tested.
- In learning:
  - At mild noise (σ=1), λ=0.95 keeps 93% of its clean return (177 vs 190) with a seed std of 21. Monte Carlo drops to 69% (149 vs 216) with a seed std of 87.
  - At σ≥3 the CIs overlap.
  - At σ=10 no λ learns reliably.

**4. Critic over/under-estimation** (`p2_value_bias.png`)
- No λ produced systematic overestimation.
- On Hopper, for λ ≥ 0.8, the mean critic bias is within ±0.5% of the realised return.
- The size of the critic's error depends more on the environment than on λ.

## Results (Phase 3): on-policy vs off-policy value estimation

`rl/ddpg.py` is DDPG and TD3 from scratch. TD3's three changes (twin-Q, target smoothing, policy delay) are separate flags. Every 50k steps, Q(s, μ(s)) is compared against the realised discounted return on deterministic evaluation episodes, as in Fujimoto et al. 2018, Fig. 1. PPO's V(s) is compared the same way. Errors are expressed as a fraction of mean|return|, so all methods share one scale. Results cover the second half of training, 5 seeds, IQM with 95% CI (`p3_value_bias_over_training.png`, `p3_value_error_summary.png`, `results/tables/p3_value_summary.csv`).

| | Hopper bias | Hopper per-state err std | Hopper return | LunarLander bias | LunarLander return |
|---|---|---|---|---|---|
| PPO (λ=0.95) | +0.1% [−0.2, +0.7] | 0.03 | 2719 | +4.5% [−4, +13] | 190 |
| DDPG | **+47%** [+33, +59] | 0.16 | 1515 | **+79%** [+58, +97] | 244 |
| DDPG + twin-Q only | −3% [−6, +4] | 0.06 | 2769 | −5% [−7, −1] | 281 |
| TD3 | −9% [−12, −3] | 0.06 | **3212** | −10% [−15, −8] | 278 |

- **DDPG overestimates heavily.** 99% of evaluated Hopper states are overestimated, and early in training the error reaches several times the true return.
- **Clipped double-Q alone removes the overestimation.** TD3's other two changes push it into mild *underestimation*.
- **Fixing the bias doubles DDPG's Hopper return:** 1515 for DDPG, 3212 for TD3.
- **PPO's critic is essentially unbiased** on both tasks. On Hopper, its per-state error is about 5× smaller than DDPG's (0.03 vs 0.16), and its across-seed spread is about 25× smaller (0.005 vs 0.13).
- **On LunarLander, PPO's per-state error (0.53) is as large as DDPG's.** PPO's V is unbiased, but it is not low-variance there; the twin-Q critics are tighter (0.13).

## Results (Phase 4): advantage magnitude vs update instability

Instability is measured per PPO update by:
- **KL(π_old‖π_new):** over the whole batch, after the update
- **clip fraction**
- **pre-clipping gradient norm**

`analysis/phase4.py` produces the figures; the tables are `results/tables/p4_*.csv`.

**Observational** (`p4_correlation_heatmap.png`, `p4_kurtosis_deciles.png`; 80 λ-sweep runs, ~39k updates, within-run Spearman ρ)
- Both series drift during training, and the LR is annealed. That alone creates correlation: raw Hopper ρ(|A|, grad norm) = −0.75 falls to −0.10 after detrending with a rolling median.
- **After detrending,** advantage size and tail-heaviness correlate only weakly with KL and clip fraction: ρ ≈ +0.05 to +0.14, positive in 70–95% of runs, with the CI over runs excluding 0. The most heavy-tailed decile of updates has only ~3–6% larger KL than the run average.
- Δreturn across an update is *not* used as an instability measure. It is confounded by regression to the mean.

**Intervention A: scale the reward** (`p4_intervention_adv_scale.png`; Hopper, return-std scaling off, 5 seeds)

| | ×0.1 | ×1 | ×10 |
|---|---|---|---|
| median \|A\| | 0.2 | 7–11 | 260–380 |
| norm OFF: updates with KL > 0.05 | 0.5% | 33% | 49% |
| norm OFF: final return | 2199 | 334 | 136 |
| norm ON: updates with KL > 0.05 | 0% | 0% | 0% |
| norm ON: final return | 2256 | 567 | 363 |

- **Without advantage normalisation, advantage magnitude causes unstable updates.** The 95th-percentile KL reaches 7.2 at ×1, and training collapses.
- **Normalisation removes the instability completely** (KL ≈ 0.01 at every scale), *but large rewards still hurt.*
- The likely cause is the critic. Its regression targets grow with the reward scale. Actor and critic share one optimiser and one global grad-norm clip (0.5), and the gradient norm is 18–109 at ×1/×10 vs 3.6 at ×0.1. So clipping shrinks the actor's step, and KL drops below the ×0.1 level (0.0075 vs 0.010).
- This is why PPO scales rewards by return std (the default config reaches 2719). It is about keeping the *critic's* targets at unit scale, not the advantages.
- The critic-gradient explanation fits the logged KL and gradient norms, but it has not been confirmed with a separate-optimiser ablation.

**Intervention B: "Safe Reward Balancing"** (`p4_intervention_safety_penalty.png`; Hopper with the Safety-Gymnasium velocity constraint, cost = 1 when speed > 0.74 m/s, trained on r − w·c with standard normalised PPO, 5 seeds)

| w | 0 | 0.5 | 2 | 10 |
|---|---|---|---|---|
| violation rate | 97% | 94% | 9.4% | 0.05% |
| task return | 2719 | 2400 | 505 | 843 |
| advantage kurtosis | 4.6 | 4.5 | 7.2 | **18.5** [11.4, 25.1] |
| 95th-pct KL | 0.015 | 0.016 | 0.018 | 0.014 |

- **The penalty makes advantages heavy-tailed.** Kurtosis rises 4× as w goes from 0 to 10. A rare constraint violation produces a large outlier advantage.
- **But with normalisation and clipping, update size does not change.** No update exceeded KL 0.05 at any w.
- **The real balancing problem is a cliff, not instability.** Between w=0.5 and w=2 the policy flips from ignoring the constraint (94% violations) to over-complying (return drops from 2400 to 505). w=10 complies fully with similar return to w=2 (843 vs 505; CIs overlap). No fixed weight gives both safety and return here, which motivates adaptive weights (Lagrangian) or dense shaping (Phase 5).

## Results (Phase 5): CBF reward shaping for safe navigation

**Environment.** `envs/safe_nav.py` (`SafeNav-v0`) is a 2D single-integrator robot with 8 random circular obstacles, a random goal, 16-ray lidar, and a 200-step limit. Collision ends the episode.

**Barrier function.** h_i(p) = (‖p − c_i‖² − R_i²)/R_i², where R_i is the obstacle radius plus a 0.1 margin.

**CBF shaping.** Train on r_task − k·Σ_i max(0, (1−α)h_i(p_t) − h_i(p_{t+1})), with α = 0.2. The penalty is dense and starts several steps before an impact; this is unit-tested.

**Baselines:**
- task reward only
- sparse collision penalty
- distance (proximity) penalty
- PPO-Lagrangian (adaptive multiplier, 5% collision budget)

**Reference:** a CBF-QP safety filter (`cbf_filter`) applied at deployment. u = 0 is always feasible for this robot, so the QP never fails.

**Protocol, fixed before the final runs:**
- Equal tuning budget per method: 4 coefficients × 3 seeds, chosen automatically (`analysis.phase5 --select`).
- 10 fresh seeds × 500k steps.
- Primary metric T_safe: the first step at which the last 100 training episodes have success ≥ 80% and collisions ≤ 5%.

| | T_safe (k steps) | training collisions | eval collision | eval collision + filter | eval success → + filter |
|---|---|---|---|---|---|
| Task only | 149 [124, 169] | 969 | 2.2% | **0%** | 0.88 → 0.91 |
| Sparse (C=3) | 131 [100, 160] | 882 | 2.0% | **0%** | 0.89 → 0.91 |
| Distance (k=0.3) | 133 [117, 155] | 841 | 2.3% | **0%** | 0.85 → 0.87 |
| **CBF shaping (k=0.3)** | 130 [115, 155] | 945 | 2.3% | **0%** | 0.88 → 0.90 |
| PPO-Lagrangian | 133 [115, 152] | **766** | 2.4% | **0%** | 0.84 → 0.88 |

- **Convergence speed: no significant difference.** CBF shaping vs sparse penalty: 0.6% faster, 95% CI [−37%, +22%]. vs task-only: 12.6% faster, CI [−12%, +27%]. All 50 runs reached T_safe.
- **Safety during training: PPO-Lagrangian is best.** It had 19% fewer training collisions than CBF shaping (766 vs 945; the CBF-vs-Lagrangian CI for the reduction is [−36%, −12%], i.e. CBF had more).
- **Why the shaped rewards do not help here.** Collision already ends the episode, which forfeits all future progress reward. This is an implicit penalty strong enough that even task-only PPO learns to avoid obstacles. The "sparse-reward pitfall" that dense shaping is meant to fix does not appear in this setup.
- **Too much shaping causes the same cliff as Phase 4.** In tuning, CBF k=10 and distance k ≥ 3 never reached T_safe: the agent becomes over-cautious (success 8–30%).
- **The CBF-QP filter at deployment is what actually delivers safety.** It gave 0 collisions for every trained policy, and success *rose* by 2–4 points, since it steers around obstacles instead of letting the policy crash. It changed about 30% of actions.

## What is measured

| Question | How | Output |
|---|---|---|
| Is GAE implemented correctly? | Unit tests vs a brute-force Σ(γλ)^l δ_{t+l}, λ=0/λ=1 limits, termination vs truncation | `tests/test_gae.py` |
| Does λ trade bias for variance in the *gradient*? | Frozen policy, 100 independent batches, all λ on the **same** trajectories; variance, bias², MSE vs a λ=1 reference | `p2_gradient_bias_variance.png` |
| How does λ affect learning? | 8 λ × 2 envs × 5 seeds, IQM + 95% bootstrap CI | `p2_learning_curves.png`, `p2_lambda_final_return.png` |
| Does λ help under noisy rewards? | Gaussian reward noise σ ∈ {0,1,3,10}; theory σ²/(1−(γλ)²) vs measured | `p2_noise_learning.png`, `p2_noise_theory.png` |
| Does λ change critic over/under-estimation? | V(s) vs realised discounted return on held-out rollouts | `p2_value_bias.png` |

## Method notes (the parts worth defending)

**GAE.** `rl/gae.py` takes two separate flags:
- `terminated`: the next state is truly terminal, so its value is 0. Don't bootstrap, and cut the recursion.
- `episode_end`: the episode ended for *any* reason, which includes time-limit truncation. Bootstrap from V(final obs), but still cut the recursion.

Mixing these two up is the most common GAE bug. It either bootstraps from the next episode's first state or treats time limits as deaths.

**Gradient probe** (`experiments/probe_gradients.py`):
1. Freeze a checkpoint and collect K batches, each made of E complete episodes from fresh resets.
   Batches are complete episodes because cutting at a step count would over-sample short episodes.
2. For every λ, compute ĝ = (1/M) Σ ∇log π(a|s) Â_λ on the same batch.
   At θ = θ_old this is exactly the gradient of PPO's clipped surrogate.
3. The λ=1 estimator is unbiased, because a state-dependent baseline adds no bias. It serves as the paired reference:
   - **Variance** = tr Cov(ĝ)
   - **Bias²** = ‖mean(ĝ_λ − ĝ_1)‖² − tr Cov(ĝ_λ − ĝ_1)/K. The second term removes the finite-K bias.
   - **Cosine** = similarity to a leave-one-out reference gradient.

   All are reported relative to ‖g*‖².
4. All λ and noise levels go through one batched backward pass (`is_grads_batched`). This is verified against plain autograd in `tests/test_ppo_smoke.py`.

**Reward-noise theory.** With i.i.d. reward noise ε ~ N(0, σ²) and a fixed critic, the noise part of Â_t is Σ_l (γλ)^l ε_{t+l}. Its variance is σ² Σ (γλ)^{2l} = σ²/(1−(γλ)²), which is ≈50× at λ=1 and ≈8.6× at λ=0.95 (γ=0.99). The probe checks this against measurement using the exact finite-horizon form.

**PPO** (`rl/ppo.py`) follows the "37 implementation details" recipe:
- orthogonal init
- state-independent log-std
- observation normalisation, and reward scaling by return std
- per-minibatch advantage normalisation
- value clipping and gradient clipping
- LR annealing

Every component is a config toggle. Each update also logs raw advantage statistics, KL, clip fraction and gradient norm; these are used in Phase 4.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest
```

```bash
./run_phase2.sh && ./run_phase3.sh && ./run_phase4.sh && ./run_phase5.sh
```

```bash
./make_figures.sh
```

The `run_phase*.sh` scripts are resumable: re-running skips finished runs and probes. `make_figures.sh` regenerates every figure and table from saved data in about 2 minutes.

Wall-clock times on an Apple M5 with 9 workers:

| Pipeline | Time |
|---|---|
| Phase 2 | ~2.5 h |
| Phase 3 | ~7 h (DDPG updates every step) |
| Phase 4 | ~45 min |
| Phase 5 | ~1.5 h |

## Layout

```
rl/              gae.py, ppo.py (+Lagrangian), ddpg.py (DDPG/TD3), networks.py, normalize.py
envs/            wrappers.py (reward noise, safety cost, vector env), safe_nav.py (SafeNav-v0 + CBF filter)
instrumentation/ probe.py     (frozen-agent episode collection, batched gradients)
experiments/     sweep.py, train_ppo.py, probe_gradients.py, probe_values.py, eval_safenav.py, configs/
analysis/        phase2.py ... phase5.py (every figure + table)
tests/
```
