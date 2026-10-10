"""Deterministic evaluation of trained SafeNav policies, with and without the CBF-QP safety filter.

Every policy is evaluated on the same 200 fixed layouts (seeds 50_000+k), using the mean action.
The filter (envs.safe_nav.SafeNavEnv.cbf_filter) minimally changes the action so the
discrete-time CBF condition holds for every obstacle; it is the "provably safe" reference.

    python -m experiments.eval_safenav results/runs/safenav_final --workers 9
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

N_EVAL = 200


def eval_run(run_dir: str) -> list[dict]:
    import gymnasium as gym
    import torch

    torch.set_num_threads(1)
    from experiments.probe_gradients import list_checkpoints
    from instrumentation.probe import load_agent

    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = load_agent(list_checkpoints(run_dir)[-1])
    kw = dict(cfg["env_kwargs"] or {})
    env = gym.make("SafeNav-v0", **kw).unwrapped
    rows = []
    for filtered in (False, True):
        succ = coll = 0
        lengths, h_mins, interventions = [], [], []
        for k in range(N_EVAL):
            obs, _ = env.reset(seed=50_000 + k)
            n_int, h_min = 0, np.inf
            for t in range(env.max_steps):
                with torch.no_grad():
                    a = agent.actor.dist(torch.as_tensor(agent.norm(obs[None]))).mean[0].numpy()
                a = np.clip(a, -1, 1)
                if filtered:
                    a_f = env.cbf_filter(a)
                    n_int += float(np.linalg.norm(a_f - a) > 1e-6)
                    a = a_f
                obs, _, term, trunc, info = env.step(a)
                h_min = min(h_min, info["h_min"])
                if term or trunc:
                    break
            succ += info["success"]
            coll += info["cost"]
            lengths.append(t + 1)
            h_mins.append(h_min)
            interventions.append(n_int / (t + 1))
        rows.append({
            "run": os.path.basename(run_dir), "method": cfg.get("method_label") or kw.get("reward_mode", "task"),
            "lagrangian": cfg.get("lagrangian", False), "coef": kw.get("coef", 0.0), "seed": cfg["seed"],
            "filtered": filtered, "success": succ / N_EVAL, "collision": coll / N_EVAL,
            "timeout": 1 - (succ + coll) / N_EVAL, "mean_length": float(np.mean(lengths)),
            "min_h_p5": float(np.percentile(h_mins, 5)), "filter_intervention_rate": float(np.mean(interventions)),
        })
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = p.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(a.sweep_dir, "*")) if os.path.exists(os.path.join(d, "done")))
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_run, runs), 1):
            rows.extend(r)
            print(f"[{i}/{len(runs)}]", flush=True)
    out = os.path.join("results", "probes", f"eval_{os.path.basename(a.sweep_dir.rstrip('/'))}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).sort_values(["method", "seed", "filtered"]).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
