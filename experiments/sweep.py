"""Run a grid of PPO configs in parallel (one process per run, resumable).

    python -m experiments.sweep experiments/configs/lambda_sweep.yaml --workers 9

A sweep YAML has a `name`, an optional `trainer` (ppo | ddpg), a `base` dict of PPOConfig overrides, and a `grid` dict of
lists; the cartesian product of the grid is run. Runs whose directory already contains
a `done` marker are skipped, so an interrupted sweep can simply be re-launched.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback

import yaml

ABBREV = {"gae_lambda": "lam", "reward_noise": "noise", "seed": "s", "env_id": "", "norm_adv": "normadv", "algo": "", "reward_multiplier": "rmul", "cost_weight": "w", "lam_lr": "lamlr"}


def run_name(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, dict):  # e.g. env_kwargs: {reward_mode: cbf, coef: 1.0} -> "cbf_coef1.0"
            parts.append("_".join(str(x) if kk == "reward_mode" else f"{kk}{x}" for kk, x in v.items()))
            continue
        tag = ABBREV.get(k, k)
        parts.append(f"{tag}{v}" if tag else str(v))
    return "_".join(parts)


def expand(spec: dict) -> list[dict]:
    keys = list(spec["grid"].keys())
    jobs = []
    for values in itertools.product(*(spec["grid"][k] for k in keys)):
        params = dict(zip(keys, values))
        cfg = dict(spec.get("base", {}))
        cfg.update(params)
        cfg["out_dir"] = os.path.join("results", "runs", spec["name"], run_name(params))
        jobs.append((spec.get("trainer", "ppo"), cfg))
    return jobs


def worker(job: tuple[str, dict]) -> tuple[str, str]:
    import torch

    torch.set_num_threads(1)
    trainer, cfg = job
    if trainer == "ddpg":
        from rl.ddpg import DDPGConfig as Config, train
    else:
        from rl.ppo import PPOConfig as Config, train

    if os.path.exists(os.path.join(cfg["out_dir"], "done")):
        return cfg["out_dir"], "skipped"
    if os.path.exists(cfg["out_dir"]):
        shutil.rmtree(cfg["out_dir"])  # partial run from an interrupted sweep
    t0 = time.time()
    try:
        train(Config(**cfg))
    except Exception:
        return cfg["out_dir"], "FAILED\n" + traceback.format_exc()
    return cfg["out_dir"], f"ok ({(time.time() - t0) / 60:.1f} min)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    with open(a.spec) as f:
        spec = yaml.safe_load(f)
    jobs = expand(spec)
    print(f"{spec['name']}: {len(jobs)} runs, {a.workers} workers", flush=True)
    if a.dry_run:
        for _, j in jobs:
            print(j["out_dir"])
        return

    failed = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (out, status) in enumerate(pool.imap_unordered(worker, jobs), 1):
            print(f"[{i}/{len(jobs)}] {out}: {status}", flush=True)
            failed += status.startswith("FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
