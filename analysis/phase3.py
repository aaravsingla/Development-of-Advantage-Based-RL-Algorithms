"""Phase 3 figures and tables: on-policy (PPO) vs off-policy (DDPG / TD3) value estimation.

    python -m analysis.phase3

Value-estimation error is always (prediction - realised discounted return) / mean|return|,
so PPO's V(s) (scaled-reward units) and DDPG's Q(s, mu(s)) (raw units) are on one scale:
    rel_bias     signed mean error           > 0 = overestimation
    rel_err_std  per-state spread of error   (how noisy individual value estimates are)
    seed_std     across-seed std of rel_bias (how reproducible the estimate is)
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.phase2 import AXIS, BIN, INK, INK2, SURFACE, TAB, binned_curve, boot_ci, iqm, save, short_env

METHODS = ["PPO (λ=0.95)", "DDPG", "DDPG + twin-Q", "TD3"]
COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # validated categorical slots 1-4
LATE = 0.5  # "late training" = second half


def method_of(cfg: dict) -> str:
    if "gae_lambda" in cfg:
        return f"PPO (λ={cfg['gae_lambda']:g})"
    if cfg["algo"] == "td3":
        return "TD3"
    return "DDPG + twin-Q" if cfg["twin_q"] else "DDPG"


def load_offpolicy() -> tuple[pd.DataFrame, pd.DataFrame]:
    vals, runs = [], []
    for d in sorted(glob.glob("results/runs/offpolicy_sweep/*") + glob.glob("results/runs/twinq_ablation/*")):
        if not os.path.exists(os.path.join(d, "done")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        v = pd.read_csv(os.path.join(d, "value_est.csv"))
        v["method"], v["env_id"], v["seed"] = method_of(cfg), cfg["env_id"], cfg["seed"]
        vals.append(v)
        runs.append({"dir": d, "method": method_of(cfg), "env_id": cfg["env_id"], "seed": cfg["seed"],
                     "total_steps": cfg["total_steps"]})
    return (pd.concat(vals) if vals else pd.DataFrame()), pd.DataFrame(runs)


def load_ppo() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = "results/probes/values_lambda_sweep.csv"
    if not os.path.exists(path):
        return pd.DataFrame(), pd.DataFrame()
    v = pd.read_csv(path)
    v = v[(v.lam == 0.95) & (v.sigma == 0)].copy()
    v["method"] = "PPO (λ=0.95)"
    v["rel_err_std"] = np.sqrt(np.maximum(v.rel_rmse**2 - v.rel_bias**2, 0))
    runs = []
    for d in sorted(glob.glob("results/runs/lambda_sweep/*_lam0.95_s*")):
        cfg = json.load(open(os.path.join(d, "config.json")))
        runs.append({"dir": d, "method": "PPO (λ=0.95)", "env_id": cfg["env_id"], "seed": cfg["seed"],
                     "total_steps": cfg["total_steps"]})
    return v, pd.DataFrame(runs)


def fig_bias_over_training(v: pd.DataFrame):
    envs = sorted(v.env_id.unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(5.6 * len(envs), 3.8), squeeze=False)
    for ax, env in zip(axes[0], envs):
        for m in METHODS:
            h = v[(v.env_id == env) & (v.method == m)].groupby("global_step")["rel_bias"].agg(list)
            if h.empty:
                continue
            x = h.index / 1e6
            mid = np.array([iqm(s) for s in h.values])
            ci = np.array([boot_ci(s, n=300) for s in h.values])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(x, mid, color=COLORS[m], label=m, marker="o" if m.startswith("PPO") else None, markersize=4)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_yscale("symlog", linthresh=0.2)  # early DDPG errors reach tens of x the return; keep small ones readable
        ax.set_title(short_env(env), loc="left")
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("(value − realised return) / mean|return|  (symlog)")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle("Value-estimation bias over training  ·  >0 = overestimation  ·  IQM over 5 seeds, 95% CI  ·  table: p3_value_summary.csv",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p3_value_bias_over_training.png")


def late_summary(v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    late = v[v.global_step >= LATE * v.groupby("env_id").global_step.transform("max")]
    per_seed = late.groupby(["env_id", "method", "seed"]).agg(
        rel_bias=("rel_bias", "mean"), rel_err_std=("rel_err_std", "mean"), frac_over=("frac_over", "mean"),
        temporal_std=("rel_bias", "std")).reset_index()
    rows = []
    for (env, m), g in per_seed.groupby(["env_id", "method"]):
        row = {"env_id": env, "method": m, "n_seeds": len(g)}
        for col in ["rel_bias", "rel_err_std", "frac_over", "temporal_std"]:
            row[col] = iqm(g[col])
            row[f"{col}_lo"], row[f"{col}_hi"] = boot_ci(g[col])
        row["seed_std"] = g.rel_bias.std()
        rows.append(row)
    return per_seed, pd.DataFrame(rows)


def fig_summary(summ: pd.DataFrame):
    envs = sorted(summ.env_id.unique())
    panels = [("rel_bias", "bias  (>0 = overestimation)"), ("rel_err_std", "per-state error std"),
              ("seed_std", "across-seed std of bias")]
    fig, axes = plt.subplots(len(envs), 3, figsize=(13, 3.2 * len(envs)), squeeze=False)
    for i, env in enumerate(envs):
        s = summ[summ.env_id == env].set_index("method").reindex([m for m in METHODS if m in set(summ.method)])
        y = np.arange(len(s))[::-1]
        for j, (col, label) in enumerate(panels):
            ax = axes[i][j]
            for k, m in enumerate(s.index):
                val = s.loc[m, col]
                if f"{col}_lo" in s:
                    ax.plot([s.loc[m, f"{col}_lo"], s.loc[m, f"{col}_hi"]], [y[k]] * 2, color=COLORS[m], linewidth=2, alpha=0.5)
                ax.plot(val, y[k], "o", color=COLORS[m], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
                ax.annotate(f"{val:+.2f}" if col == "rel_bias" else f"{val:.2f}", (val, y[k]), xytext=(0, 8),
                            textcoords="offset points", ha="center", color=INK2, fontsize=8)
            if col == "rel_bias":
                ax.axvline(0, color=AXIS, linewidth=1)
            ax.set_yticks(y, s.index if j == 0 else [""] * len(s))
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{short_env(env)} · {label}" if j == 0 else label, loc="left", fontsize=10)
            ax.set_xlabel("fraction of mean|return|")
    fig.suptitle("Value estimation in the second half of training  ·  IQM over 5 seeds, 95% CI",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p3_value_error_summary.png")


def fig_learning(runs: pd.DataFrame) -> pd.DataFrame:
    envs = sorted(runs.env_id.unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(5.6 * len(envs), 3.6), squeeze=False)
    finals = []
    for ax, env in zip(axes[0], envs):
        for m in METHODS:
            g = runs[(runs.env_id == env) & (runs.method == m)]
            if g.empty:
                continue
            curves = np.stack([binned_curve(r.dir, r.total_steps) for _, r in g.iterrows()])
            x = (np.arange(curves.shape[1]) + 1) * BIN / 1e6
            mid = np.array([iqm(curves[:, t]) for t in range(curves.shape[1])])
            ci = np.array([boot_ci(curves[:, t], n=300) for t in range(curves.shape[1])])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(x, mid, color=COLORS[m], label=m)
            for _, r in g.iterrows():
                ep = pd.read_csv(os.path.join(r.dir, "episodes.csv"))
                finals.append({"env_id": env, "method": m, "seed": r.seed,
                               "final": ep[ep.global_step > 0.9 * r.total_steps]["return"].mean()})
        ax.set_title(short_env(env), loc="left")
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("training episode return (IQM, 95% CI)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Learning curves: on-policy vs off-policy  ·  5 seeds", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p3_learning_curves.png")
    f = pd.DataFrame(finals)
    return f.groupby(["env_id", "method"])["final"].agg(final_iqm=iqm, n="count").reset_index()


def main():
    os.makedirs(TAB, exist_ok=True)
    v_off, runs_off = load_offpolicy()
    v_ppo, runs_ppo = load_ppo()
    cols = ["env_id", "method", "seed", "global_step", "rel_bias", "rel_err_std", "frac_over"]
    v = pd.concat([x[cols] for x in (v_ppo, v_off) if not x.empty])
    v = v.dropna(subset=["rel_bias"])
    fig_bias_over_training(v)
    per_seed, summ = late_summary(v)
    finals = fig_learning(pd.concat([runs_ppo, runs_off]))
    summ = summ.merge(finals, on=["env_id", "method"], how="left")
    summ.to_csv(os.path.join(TAB, "p3_value_summary.csv"), index=False)
    per_seed.to_csv(os.path.join(TAB, "p3_value_per_seed.csv"), index=False)
    fig_summary(summ)
    print(summ[["env_id", "method", "n_seeds", "final_iqm", "rel_bias", "rel_bias_lo", "rel_bias_hi", "rel_err_std",
                "seed_std", "frac_over"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
