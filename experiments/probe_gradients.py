"""Direct measurement of policy-gradient bias and variance as a function of GAE lambda.

For a frozen checkpoint we draw K independent batches (each = E complete episodes). On
every batch we compute the policy gradient once per (lambda, reward-noise sigma) using the
*same* trajectories. Reference: lambda = 1 without noise, which is unbiased for the true
policy gradient (a state-dependent baseline adds no bias). Then, per estimator:

    variance   tr Cov(g_hat)                         spread of the estimate across batches
    bias^2     ||E[g_hat - g_hat^{lam=1}]||^2         paired differences, de-biased for finite K
    mse        bias^2 + variance                      expected squared error of a single batch

All three are reported relative to the variance of the Monte Carlo (lambda=1) estimator,
so lambda=1 sits at exactly 1.0 and "mse_ratio = 0.2" reads "5x lower error than Monte Carlo".
(Normalising by ||true gradient||^2 instead is unstable: late in training the MC gradient's
per-batch variance is hundreds of times ||g||^2, so ||g||^2 itself cannot be resolved from K
batches. `snr_mc` = ||g||^2 / Var_MC is kept in the CSV to show exactly this.)
95% CIs come from bootstrapping over batches.

    python -m experiments.probe_gradients results/runs/lambda_sweep/Hopper-v5_lam0.95_s0 --batches 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from instrumentation.probe import (
    EpisodeCollector,
    batched_policy_gradients,
    episode_advantages,
    episode_values,
    load_agent,
)
from rl.gae import reward_noise_variance_multiplier

LAMBDAS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0]


def list_checkpoints(run_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(run_dir, "ckpt_*.pt"))
    return sorted(paths, key=lambda p: int(re.search(r"ckpt_(\d+)", p).group(1)))


def pick_checkpoints(run_dir: str, fractions: list[float]) -> list[str]:
    ckpts = list_checkpoints(run_dir)
    return [ckpts[min(len(ckpts) - 1, int(round(f * (len(ckpts) - 1))))] for f in fractions]


def estimator_stats(G: np.ndarray, G_ref: np.ndarray) -> dict:
    """G, G_ref: [K, P] per-batch gradients of the estimator and of the paired lambda=1 reference."""
    K = G.shape[0]
    var = G.var(axis=0, ddof=1).sum()
    d = G - G_ref
    bias2 = max(0.0, float((d.mean(0) ** 2).sum() - d.var(axis=0, ddof=1).sum() / K))
    var_mc = G_ref.var(axis=0, ddof=1).sum()
    norm2_true = float((G_ref.mean(0) ** 2).sum() - var_mc / K)
    loo = (G_ref.sum(0, keepdims=True) - G_ref) / (K - 1)
    cos = (G * loo).sum(1) / (np.linalg.norm(G, axis=1) * np.linalg.norm(loo, axis=1) + 1e-12)
    return {
        "var": var,
        "bias2": bias2,
        "mse": bias2 + var,
        "var_mc": var_mc,
        "var_ratio": var / var_mc,
        "bias2_ratio": bias2 / var_mc,
        "mse_ratio": (bias2 + var) / var_mc,
        "snr_mc": norm2_true / var_mc,
        "cos_mean": float(cos.mean()),
    }


def bootstrap_ci(G: np.ndarray, G_ref: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    K = G.shape[0]
    keys = ["var_ratio", "bias2_ratio", "mse_ratio"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, K, K)
        s = estimator_stats(G[idx], G_ref[idx])
        for k in keys:
            samples[k].append(s[k])
    out = {}
    for k in keys:
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(samples[k], [2.5, 97.5])
    return out


def probe_checkpoint(ckpt: str, n_batches: int, episodes_per_batch: int, lambdas: list[float],
                     sigmas: list[float], seed: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    agent = load_agent(ckpt)
    collector = EpisodeCollector(agent.env_id, episodes_per_batch)
    configs = [(lam, s) for s in sigmas for lam in lambdas]
    ref_idx = configs.index((1.0, 0.0))
    grads = [[] for _ in configs]
    noise_rng = np.random.default_rng(seed + 12345)

    adv_rows = []  # advantage-level statistics, for checking the reward-noise variance formula
    batch_sizes = []
    for b in range(n_batches):
        eps = collector.collect(agent, seed=seed * 100_000 + b)
        obs = np.concatenate([e.obs for e in eps])
        act = np.concatenate([e.act for e in eps])
        M = len(obs)
        batch_sizes.append(M)
        vals = [episode_values(agent, e) for e in eps]
        noises = {s: [s * noise_rng.standard_normal(len(e.rew)) for e in eps] for s in sigmas}

        W = np.zeros((len(configs), M))
        clean = {}
        for c, (lam, s) in enumerate(configs):
            adv = np.concatenate([
                episode_advantages(agent, e, v, vf, lam, noises[s][i] if s > 0 else None)
                for i, (e, (v, vf)) in enumerate(zip(eps, vals))
            ])
            W[c] = adv / M
            if s == 0:
                clean[lam] = adv
            elif b < 20:
                # Exact finite-horizon theory: with H steps left in the episode, the noise part of
                # A_t is sum_{l<H} (gamma*lam)^l eps_{t+l} / scale, with variance
                # (sigma/scale)^2 * (1 - (gamma*lam)^(2H)) / (1 - (gamma*lam)^2).
                gl2 = (agent.gamma * lam) ** 2
                H = np.concatenate([np.arange(len(e.rew), 0, -1) for e in eps])
                per_step = (H if gl2 == 1.0 else (1 - gl2**H) / (1 - gl2))
                adv_rows.append({
                    "lam": lam, "sigma": s,
                    "noise_var_measured": float(((adv - clean[lam]) ** 2).mean()),
                    "noise_var_theory": float((s / agent.reward_scale) ** 2 * per_step.mean()),
                    "infinite_horizon_multiplier": reward_noise_variance_multiplier(agent.gamma, lam),
                })
        G = batched_policy_gradients(agent.actor, obs, act, W)
        for c in range(len(configs)):
            grads[c].append(G[c])
    collector.close()

    grads = [np.stack(g) for g in grads]
    rng = np.random.default_rng(seed)
    rows = []
    for c, (lam, s) in enumerate(configs):
        row = {"checkpoint": os.path.basename(ckpt), "global_step": agent.global_step, "lam": lam, "sigma": s,
               "n_batches": n_batches, "mean_batch_size": float(np.mean(batch_sizes))}
        row.update(estimator_stats(grads[c], grads[ref_idx]))
        row.update(bootstrap_ci(grads[c], grads[ref_idx], n_boot, rng))
        rows.append(row)
    adv_df = pd.DataFrame(adv_rows)
    if len(adv_df):
        adv_df = adv_df.groupby(["lam", "sigma"], as_index=False).mean()
        adv_df["global_step"] = agent.global_step
    return pd.DataFrame(rows), adv_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.4, 1.0],
                   help="which checkpoints, as fractions of training (early / mid / late)")
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--episodes-per-batch", type=int, default=8)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0])
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.set_num_threads(1)
    out = a.out or os.path.join("results", "probes", "grad_" + os.path.basename(a.run_dir.rstrip("/")))
    os.makedirs(out, exist_ok=True)
    run_cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"run_dir": a.run_dir, "env_id": run_cfg["env_id"], "seed": run_cfg["seed"],
                   "lam_trained": run_cfg["gae_lambda"], "args": vars(a)}, f, indent=2)
    all_stats, all_adv = [], []
    for ckpt in pick_checkpoints(a.run_dir, a.fractions):
        t0 = time.time()
        stats, adv = probe_checkpoint(ckpt, a.batches, a.episodes_per_batch, a.lambdas, a.sigmas, a.seed, a.boot)
        all_stats.append(stats)
        all_adv.append(adv)
        print(f"{ckpt}: {time.time() - t0:.0f}s, batch ~{stats.mean_batch_size.iloc[0]:.0f} steps")
        print(stats[["lam", "sigma", "var_ratio", "bias2_ratio", "mse_ratio", "snr_mc"]].to_string(index=False), flush=True)
    pd.concat(all_stats).to_csv(os.path.join(out, "grad_stats.csv"), index=False)
    if any(len(x) for x in all_adv):
        pd.concat(all_adv).to_csv(os.path.join(out, "adv_noise_stats.csv"), index=False)


if __name__ == "__main__":
    main()
