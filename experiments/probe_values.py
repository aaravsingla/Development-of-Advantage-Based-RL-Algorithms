"""Critic accuracy vs realised returns, for every checkpoint of every run in a sweep.

For each state visited by the (stochastic) policy we compare the critic's prediction V(s_t)
with the realised discounted return G_t = sum_k gamma^k r_{t+k} (in the critic's scaled-reward
units). Averaged over many states, V - G estimates the critic's bias:
    > 0  overestimation,  < 0  underestimation.

Time-limit truncation: G_t is only a valid target when the episode actually terminated, or
when enough steps remain that the cut-off tail is negligible. We keep states with at least
H = log(0.01)/log(gamma) (~459 for gamma=0.99) steps left before a truncation, so the
missing tail weighs < 1% of the return.

    python -m experiments.probe_values results/runs/lambda_sweep --workers 9
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os

import numpy as np
import pandas as pd


def probe_run(args) -> list[dict]:
    run_dir, every, n_episodes, seed = args
    import torch

    torch.set_num_threads(1)
    from experiments.probe_gradients import list_checkpoints
    from instrumentation.probe import EpisodeCollector, discounted_mc_returns, episode_values, load_agent

    rows = []
    ckpts = list_checkpoints(run_dir)
    ckpts = ckpts[every - 1 :: every] if every > 1 else ckpts
    collector = None
    for ck in ckpts:
        agent = load_agent(ck)
        if collector is None:
            collector = EpisodeCollector(agent.env_id, n_episodes)
        H = int(np.ceil(np.log(0.01) / np.log(agent.gamma)))
        errs, gs = [], []
        for ep in collector.collect(agent, seed=seed):
            v, _ = episode_values(agent, ep)
            g = discounted_mc_returns(agent.scale_rewards(ep.rew), agent.gamma)
            keep = np.ones(len(g), bool) if ep.terminated else np.arange(len(g)) < len(g) - H
            errs.append((v - g)[keep])
            gs.append(g[keep])
        e, g = np.concatenate(errs), np.concatenate(gs)
        if len(e) == 0:
            continue
        cfg = agent.config
        rows.append({
            "run": os.path.basename(run_dir), "env_id": cfg["env_id"], "lam": cfg["gae_lambda"],
            "seed": cfg["seed"], "sigma": cfg["reward_noise"], "global_step": agent.global_step,
            "n_states": len(e), "bias": e.mean(), "rmse": np.sqrt((e**2).mean()),
            "mean_abs_return": np.abs(g).mean(), "rel_bias": e.mean() / np.abs(g).mean(),
            "rel_rmse": np.sqrt((e**2).mean()) / np.abs(g).mean(),
            "frac_over": float((e > 0).mean()), "reward_scale": agent.reward_scale,
        })
    if collector:
        collector.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir")
    p.add_argument("--every", type=int, default=2, help="probe every k-th checkpoint")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--out", default=None)
    a = p.parse_args()

    runs = sorted(d for d in glob.glob(os.path.join(a.sweep_dir, "*")) if os.path.exists(os.path.join(d, "done")))
    jobs = [(r, a.every, a.episodes, 777) for r in runs]
    print(f"{len(runs)} runs", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(probe_run, jobs), 1):
            rows.extend(r)
            print(f"[{i}/{len(jobs)}]", flush=True)
    out = a.out or os.path.join("results", "probes", f"values_{os.path.basename(a.sweep_dir.rstrip('/'))}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).sort_values(["env_id", "lam", "seed", "global_step"]).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
