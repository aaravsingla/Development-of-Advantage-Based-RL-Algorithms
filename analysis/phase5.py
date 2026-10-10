"""Phase 5: safe navigation, CBF reward shaping vs baselines.

    python -m analysis.phase5 --select     # after the tuning sweeps: pick each method's coefficient,
                                           # write experiments/configs/safenav_final*.yaml
    python -m analysis.phase5              # after the final sweeps + eval: figures and tables

Metrics (fixed before any final run):
    T_safe      PRIMARY. First env step at which, over the last 100 training episodes,
                success >= 80% AND collision <= 5%. Runs that never get there are censored
                at the budget (counted as the budget; the number that reached it is reported).
    T_succ      same, success >= 80% only (ignores safety).
    train_coll  total collisions during training (safety *while learning*).
    eval_*      deterministic evaluation on 200 fixed layouts, with / without the CBF-QP filter.
Tuning: every method gets the same budget, 4 coefficients x 3 seeds (100-102). Selection by
median T_safe, ties broken by fewer training collisions. Final: 10 fresh seeds (0-9).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis.phase2 import AXIS, INK, INK2, SURFACE, TAB, boot_ci, iqm, save

WINDOW = 100
SUCC_THR, COLL_THR = 0.8, 0.05
METHODS = ["task", "sparse", "distance", "cbf", "lagrangian"]
LABEL = {"task": "Task only", "sparse": "Sparse penalty", "distance": "Distance penalty", "cbf": "CBF shaping",
         "lagrangian": "PPO-Lagrangian"}
# validated categorical slots 1-5 (blue, orange, aqua, yellow, magenta); CBF gets the strongest (blue)
COLORS = {"cbf": "#2a78d6", "sparse": "#eb6834", "distance": "#1baf7a", "lagrangian": "#eda100", "task": "#e87ba4"}
TUNE_COEFS = {"sparse": "coef", "distance": "coef", "cbf": "coef", "lagrangian": "lam_lr"}


def method_of(cfg: dict) -> str:
    return "lagrangian" if cfg.get("lagrangian") else (cfg.get("env_kwargs") or {}).get("reward_mode", "task")


def coef_of(cfg: dict) -> float:
    m = method_of(cfg)
    return cfg["lam_lr"] if m == "lagrangian" else (cfg.get("env_kwargs") or {}).get("coef", 0.0)


def run_metrics(d: str) -> dict:
    cfg = json.load(open(os.path.join(d, "config.json")))
    ep = pd.read_csv(os.path.join(d, "episodes.csv"))
    rs = ep.success.rolling(WINDOW).mean()
    rc = ep.cost.rolling(WINDOW).mean()
    budget = cfg["total_steps"]
    ok_safe = (rs >= SUCC_THR) & (rc <= COLL_THR)
    ok_succ = rs >= SUCC_THR
    t_safe = float(ep.global_step[ok_safe].iloc[0]) if ok_safe.any() else float(budget)
    t_succ = float(ep.global_step[ok_succ].iloc[0]) if ok_succ.any() else float(budget)
    late = ep[ep.global_step > 0.9 * budget]
    return {"dir": d, "method": method_of(cfg), "coef": coef_of(cfg), "seed": cfg["seed"], "budget": budget,
            "T_safe": t_safe, "reached_safe": bool(ok_safe.any()), "T_succ": t_succ, "reached_succ": bool(ok_succ.any()),
            "train_coll": float(ep.cost.sum()), "train_episodes": len(ep),
            "final_success": late.success.mean(), "final_collision": late.cost.mean(),
            "success_auc": float(ep.success.mean())}


def load(sweeps: list[str]) -> pd.DataFrame:
    dirs = [d for s in sweeps for d in sorted(glob.glob(f"results/runs/{s}/*")) if os.path.exists(os.path.join(d, "done"))]
    return pd.DataFrame([run_metrics(d) for d in dirs])


# ----------------------------------------------------------------------------- tuning
def select():
    t = load(["safenav_tune", "safenav_tune_lag"])
    s = t.groupby(["method", "coef"]).agg(T_safe_median=("T_safe", "median"), reached=("reached_safe", "sum"),
                                          n=("seed", "count"), train_coll=("train_coll", "mean"),
                                          final_success=("final_success", "mean"),
                                          final_collision=("final_collision", "mean")).reset_index()
    s = s.sort_values(["method", "T_safe_median", "train_coll"])
    os.makedirs(TAB, exist_ok=True)
    s.to_csv(os.path.join(TAB, "p5_tuning.csv"), index=False)
    print(s.round(3).to_string(index=False))
    best = s.groupby("method").head(1).set_index("method")
    print("\nselected:\n", best[["coef", "T_safe_median", "reached", "train_coll"]])

    budget = int(t.budget.iloc[0])
    kwargs = [{"reward_mode": "task", "coef": 0.0}]
    for m in ["sparse", "distance", "cbf"]:
        kwargs.append({"reward_mode": m, "coef": float(best.loc[m, "coef"])})
    final = {"name": "safenav_final", "base": {"env_id": "SafeNav-v0", "total_steps": budget},
             "grid": {"env_kwargs": kwargs, "seed": list(range(10))}}
    lag = {"name": "safenav_final", "base": {"env_id": "SafeNav-v0", "total_steps": budget, "lagrangian": True,
                                             "env_kwargs": {"reward_mode": "task", "coef": 0.0}},
           "grid": {"lam_lr": [float(best.loc["lagrangian", "coef"])], "seed": list(range(10))}}
    hdr = "# GENERATED by `python -m analysis.phase5 --select` from the tuning sweeps. Do not edit by hand.\n"
    with open("experiments/configs/safenav_final.yaml", "w") as f:
        f.write(hdr + yaml.safe_dump(final, sort_keys=False))
    with open("experiments/configs/safenav_final_lag.yaml", "w") as f:
        f.write(hdr + yaml.safe_dump(lag, sort_keys=False))
    print("wrote experiments/configs/safenav_final.yaml and safenav_final_lag.yaml")


# ----------------------------------------------------------------------------- final
def rolling_curves(runs: pd.DataFrame, grid: np.ndarray) -> dict:
    out = {}
    for m, g in runs.groupby("method"):
        S, C = [], []
        for d in g.dir:
            ep = pd.read_csv(os.path.join(d, "episodes.csv"))
            rs, rc = ep.success.rolling(WINDOW, min_periods=10).mean(), ep.cost.rolling(WINDOW, min_periods=10).mean()
            idx = np.clip(np.searchsorted(ep.global_step.values, grid), 0, len(ep) - 1)
            S.append(rs.values[idx])
            C.append(rc.values[idx])
        out[m] = (np.array(S), np.array(C))
    return out


def fig_curves(runs: pd.DataFrame):
    budget = runs.budget.iloc[0]
    grid = np.linspace(budget / 50, budget, 50)
    curves = rolling_curves(runs, grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for m in METHODS:
        if m not in curves:
            continue
        for ax, arr in zip(axes, curves[m]):
            mid = np.array([iqm(arr[:, k][~np.isnan(arr[:, k])]) for k in range(len(grid))])
            ci = np.array([boot_ci(arr[:, k][~np.isnan(arr[:, k])], n=300) for k in range(len(grid))])
            ax.fill_between(grid / 1e6, ci[:, 0], ci[:, 1], color=COLORS[m], alpha=0.13, linewidth=0)
            ax.plot(grid / 1e6, mid, color=COLORS[m], label=LABEL[m], linewidth=2.4 if m == "cbf" else 1.8)
    axes[0].axhline(SUCC_THR, color=AXIS, linestyle="--", linewidth=1)
    axes[1].axhline(COLL_THR, color=AXIS, linestyle="--", linewidth=1)
    axes[0].set_ylabel(f"success rate (last {WINDOW} episodes)")
    axes[1].set_ylabel(f"collision rate (last {WINDOW} episodes)")
    axes[0].set_title("Task success during training", loc="left")
    axes[1].set_title("Collisions during training", loc="left")
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
    axes[0].legend(fontsize=8)
    fig.suptitle("SafeNav · 10 seeds · IQM, 95% CI  (dashed: thresholds used for T_safe)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p5_training_curves.png")


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [x for x in METHODS if x in set(runs.method)]:
        g = runs[runs.method == m]
        row = {"method": m, "coef": g.coef.iloc[0], "n": len(g), "reached_safe": int(g.reached_safe.sum())}
        for col in ["T_safe", "T_succ", "train_coll", "final_success", "final_collision"]:
            row[col] = iqm(g[col])
            row[f"{col}_lo"], row[f"{col}_hi"] = boot_ci(g[col])
        rows.append(row)
    return pd.DataFrame(rows)


def ratio_ci(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """1 - IQM(a)/IQM(b): fractional reduction of a vs b, bootstrap over seeds of each independently."""
    rng = np.random.default_rng(seed)
    point = 1 - iqm(a) / iqm(b)
    s = [1 - iqm(a[rng.integers(0, len(a), len(a))]) / iqm(b[rng.integers(0, len(b), len(b))]) for _ in range(n)]
    lo, hi = np.percentile(s, [2.5, 97.5])
    return point, lo, hi


def fig_summary(summ: pd.DataFrame, ev: pd.DataFrame | None):
    n_pan = 3 if ev is not None else 2
    fig, axes = plt.subplots(1, n_pan, figsize=(5.0 * n_pan, 3.6))
    order = [m for m in METHODS if m in set(summ.method)]
    y = np.arange(len(order))[::-1]
    s = summ.set_index("method").reindex(order)
    for ax, col, title, scale in [(axes[0], "T_safe", "Steps to safe competence (T_safe)", 1e6),
                                  (axes[1], "train_coll", "Collisions during training", 1)]:
        for k, m in enumerate(order):
            ax.plot([s.loc[m, f"{col}_lo"] / scale, s.loc[m, f"{col}_hi"] / scale], [y[k]] * 2, color=COLORS[m], linewidth=2, alpha=0.5)
            ax.plot(s.loc[m, col] / scale, y[k], "o", color=COLORS[m], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
            txt = f"{s.loc[m, col] / scale:.2f}M" if col == "T_safe" else f"{s.loc[m, col]:.0f}"
            if col == "T_safe":
                txt += f"  ({s.loc[m, 'reached_safe']}/{s.loc[m, 'n']} reached)"
            ax.annotate(txt, (s.loc[m, col] / scale, y[k]), xytext=(0, 8), textcoords="offset points", ha="center",
                        color=INK2, fontsize=8)
        ax.set_yticks(y, [LABEL[m] for m in order] if ax is axes[0] else [""] * len(order))
        ax.set_ylim(-0.6, len(order) - 0.2)  # headroom for the value labels
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_xlabel("env steps (millions; censored at budget)")
    axes[1].set_xlabel("collisions (IQM over seeds)")
    if ev is not None:
        ax = axes[2]
        e = ev.groupby(["method", "filtered"]).collision.agg(lambda x: iqm(x)).unstack().reindex(order)
        sc = ev.groupby(["method", "filtered"]).success.agg(lambda x: iqm(x)).unstack().reindex(order)
        for k, m in enumerate(order):
            ax.plot([e.loc[m, True], e.loc[m, False]], [y[k]] * 2, color=AXIS, linewidth=1.5, zorder=1)
            ax.plot(e.loc[m, False], y[k], "o", color=COLORS[m], markersize=8, markeredgecolor=SURFACE, zorder=3)
            ax.plot(e.loc[m, True], y[k], "D", color=COLORS[m], markersize=6, markerfacecolor=SURFACE, zorder=3)
            ax.annotate(f"succ {sc.loc[m, False]:.2f} → {sc.loc[m, True]:.2f}", (max(e.loc[m, False], e.loc[m, True]), y[k]),
                        xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=7.5)
        ax.set_yticks(y, [""] * len(order))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("evaluation collision rate (● policy alone, ◇ + CBF-QP filter)")
        ax.set_title("Deployment: 200 fixed layouts, deterministic", loc="left", fontsize=10)
    fig.suptitle("SafeNav results · 10 seeds · IQM, 95% CI", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p5_summary.png")


def final():
    runs = load(["safenav_final"])
    runs.to_csv(os.path.join(TAB, "p5_final_per_seed.csv"), index=False)
    summ = summarise(runs)
    summ.to_csv(os.path.join(TAB, "p5_final_summary.csv"), index=False)
    print(summ[["method", "coef", "n", "reached_safe", "T_safe", "T_safe_lo", "T_safe_hi", "T_succ", "train_coll",
                "final_success", "final_collision"]].round(3).to_string(index=False))
    head = []
    cbf = runs[runs.method == "cbf"]
    for base in ["sparse", "distance", "lagrangian", "task"]:
        b = runs[runs.method == base]
        if b.empty or cbf.empty:
            continue
        for col in ["T_safe", "train_coll"]:
            p, lo, hi = ratio_ci(cbf[col].values, b[col].values)
            head.append({"vs": base, "metric": col, "reduction": p, "lo": lo, "hi": hi})
    head = pd.DataFrame(head)
    head.to_csv(os.path.join(TAB, "p5_headline.csv"), index=False)
    print("\nCBF shaping, fractional reduction vs baseline (1 - IQM_cbf / IQM_base), 95% CI:")
    print(head.round(3).to_string(index=False))
    fig_curves(runs)
    evf = "results/probes/eval_safenav_final.csv"
    ev = pd.read_csv(evf) if os.path.exists(evf) else None
    if ev is not None:
        ev["method"] = np.where(ev.lagrangian, "lagrangian", ev.method)
        es = ev.groupby(["method", "filtered"])[["success", "collision", "timeout", "filter_intervention_rate"]].agg(
            lambda x: iqm(x)).reset_index()
        es.to_csv(os.path.join(TAB, "p5_eval_summary.csv"), index=False)
        print("\nevaluation (IQM over seeds):")
        print(es.round(3).to_string(index=False))
    fig_summary(summ, ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    a = p.parse_args()
    os.makedirs(TAB, exist_ok=True)
    select() if a.select else final()


if __name__ == "__main__":
    main()
