"""Phase 4: advantage magnitude/shape vs policy-update instability.

    python -m analysis.phase4

Part 1 (observational, 80 lambda-sweep runs, ~39k updates):
    within-run Spearman correlation between raw-advantage statistics and instability metrics.
    Both drift with training (and the learning rate is annealed), which alone would create a
    correlation, so each series is also detrended (minus a centred rolling median, 21 updates)
    before correlating. Reported: median rho over runs with a bootstrap CI over runs.

Part 2 (interventions):
    A  reward multiplier x {0.1, 1, 10}, advantage normalisation on/off  (adv_scale sweep)
    B  safety-penalty weight w on a velocity constraint                   (safety_penalty sweep)

Instability metrics per update:
    kl_full    KL(pi_old || pi_new) on the whole batch after the update
    clipfrac   fraction of samples whose ratio left [1-eps, 1+eps]
    grad_norm  pre-clipping gradient norm (mean over minibatches)
    d_return   change in mean episode return from the rollout before to the rollout after.
               Kept in the CSVs but NOT used as an instability measure: it is confounded by
               regression to the mean (a rollout with unusually bad episodes produces large
               advantages, and the next rollout recovers regardless of the update).
    spike      kl_full > 0.05 (2.5x the usual 0.02 KL target)
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from analysis.phase2 import AXIS, CAT, INK, INK2, LAMBDA_RAMP, MUTED, SURFACE, TAB, boot_ci, iqm, save, short_env

ADV = ["adv_mean_abs", "adv_std", "adv_p99_abs", "adv_max_abs", "adv_kurtosis"]
INST_ALL = ["kl_full", "clipfrac", "grad_norm", "d_return"]
INST = ["kl_full", "clipfrac", "grad_norm"]  # figures
LABEL = {"adv_mean_abs": "mean |A|", "adv_std": "std A", "adv_p99_abs": "p99 |A|", "adv_max_abs": "max |A|",
         "adv_kurtosis": "kurtosis A", "kl_full": "KL", "clipfrac": "clip frac", "grad_norm": "grad norm",
         "d_return": "Δ return"}
SPIKE_KL = 0.05
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])  # blue - gray - red


def load_sweep(name: str) -> list[tuple[dict, pd.DataFrame]]:
    out = []
    for d in sorted(glob.glob(f"results/runs/{name}/*")):
        if not os.path.exists(os.path.join(d, "done")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        u = pd.read_csv(os.path.join(d, "updates.csv"))
        ep = pd.read_csv(os.path.join(d, "episodes.csv"))
        # mean return of episodes finishing in each update's rollout window
        edges = np.concatenate([[0], u.global_step.values])
        idx = np.searchsorted(edges, ep.global_step.values, side="left") - 1
        per_window = pd.Series(ep["return"].values).groupby(idx).mean()
        win = per_window.reindex(range(len(u))).ffill().values
        u["d_return"] = np.append(win[1:] - win[:-1], np.nan)  # update u happens between window u and u+1
        u["spike"] = (u.kl_full > SPIKE_KL).astype(float)
        u["dir"] = d
        if "cost" in ep:
            u.attrs["episodes"] = ep
        out.append((cfg, u))
    return out


def detrend(x: pd.Series, w: int = 21) -> pd.Series:
    return x - x.rolling(w, center=True, min_periods=5).median()


def correlations(runs) -> pd.DataFrame:
    rows = []
    for cfg, u in runs:
        u = u.iloc[5:]  # skip the first updates (normaliser warm-up)
        for mode in ["raw", "detrended"]:
            for a in ADV:
                for b in INST_ALL:
                    x, y = u[a], u[b]
                    if mode == "detrended":
                        x, y = detrend(x), detrend(y)
                    ok = x.notna() & y.notna()
                    if ok.sum() < 30:
                        continue
                    rho = spearmanr(x[ok], y[ok]).statistic
                    rows.append({"env_id": cfg["env_id"], "lam": cfg["gae_lambda"], "seed": cfg["seed"], "mode": mode,
                                 "adv": a, "inst": b, "rho": rho})
    return pd.DataFrame(rows)


def fig_corr_heatmap(c: pd.DataFrame):
    envs = sorted(c.env_id.unique())
    fig, axes = plt.subplots(2, len(envs), figsize=(4.6 * len(envs), 7.4), squeeze=False)
    for i, mode in enumerate(["raw", "detrended"]):
        for j, env in enumerate(envs):
            ax = axes[i][j]
            g = c[(c.env_id == env) & (c["mode"] == mode) & c.inst.isin(INST)]
            M = g.groupby(["adv", "inst"]).rho.median().unstack().reindex(index=ADV, columns=INST)
            ax.imshow(M.values, cmap=DIVERGING, vmin=-0.6, vmax=0.6, aspect="auto")
            for r in range(M.shape[0]):
                for k in range(M.shape[1]):
                    sub = g[(g.adv == ADV[r]) & (g.inst == INST[k])].rho
                    lo, hi = boot_ci(sub, stat=np.median, n=500)
                    sig = (lo > 0) or (hi < 0)
                    ax.text(k, r, f"{M.values[r, k]:+.2f}" + ("" if sig else "\n(n.s.)"), ha="center", va="center",
                            fontsize=8.5 if sig else 7.5, color=INK if sig else MUTED)
            ax.set_xticks(range(len(INST)), [LABEL[x] for x in INST])
            ax.set_yticks(range(len(ADV)), [LABEL[x] for x in ADV])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{short_env(env)} · {mode}", loc="left")
    fig.suptitle("Within-run Spearman ρ, advantage stats vs update instability  ·  median over 40 runs per env\n"
                 "(n.s. = 95% bootstrap CI over runs includes 0;  detrended = rolling-median removed)",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    save(fig, "p4_correlation_heatmap.png")


def fig_binned(runs):
    """KL and clip fraction by within-run decile of advantage kurtosis (detrended)."""
    envs = sorted({cfg["env_id"] for cfg, _ in runs})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for color, env in zip([CAT["variance"], CAT["bias²"]], envs):
        per_run = {"kl_full": [], "clipfrac": []}
        for cfg, u in runs:
            if cfg["env_id"] != env:
                continue
            u = u.iloc[5:]
            x = detrend(u.adv_kurtosis)
            ok = x.notna()
            dec = pd.qcut(x[ok].rank(method="first"), 10, labels=False)
            for m in per_run:  # relative to the run's own mean: removes between-run differences
                vals = u.loc[ok, m]
                per_run[m].append((vals / vals.mean()).groupby(dec.values).mean().values)
        for ax, m in zip(axes, per_run):
            arr = np.stack(per_run[m])
            mid = arr.mean(0)
            ci = np.array([boot_ci(arr[:, k], stat=np.mean, n=500) for k in range(10)])
            x = np.arange(1, 11)
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mid, color=color, marker="o", markersize=5, markeredgecolor=SURFACE, label=short_env(env))
    for ax, m in zip(axes, ["KL per update ÷ run mean", "clip fraction ÷ run mean"]):
        ax.set_xlabel("decile of advantage kurtosis within run (detrended)")
        ax.set_ylabel(m + "  (95% CI over runs)")
        ax.set_xticks(range(1, 11))
        ax.axhline(1, color=AXIS, linewidth=1)
    axes[0].legend()
    axes[0].set_title("Update size by advantage tail-heaviness", loc="left")
    axes[1].set_title("Clipping by advantage tail-heaviness", loc="left")
    fig.tight_layout()
    save(fig, "p4_kurtosis_deciles.png")


def run_summary(runs, keys: list[str]) -> pd.DataFrame:
    rows = []
    for cfg, u in runs:
        ep = pd.read_csv(os.path.join(u.dir.iloc[0], "episodes.csv"))
        late_ep = ep[ep.global_step > 0.9 * cfg["total_steps"]]
        row = {k: cfg[k] for k in keys}
        row.update({
            "seed": cfg["seed"],
            "final_return": late_ep["return"].mean(),
            "adv_mean_abs": u.adv_mean_abs.median(),
            "adv_kurtosis": u.adv_kurtosis.median(),
            "kl_median": u.kl_full.median(),
            "kl_p95": u.kl_full.quantile(0.95),
            "spike_frac": u.spike.mean(),
            "clipfrac": u.clipfrac.median(),
            "grad_norm": u.grad_norm.median(),
        })
        if "cost" in ep:
            row["cost_rate"] = (late_ep.cost.sum() / late_ep.length.sum()) if len(late_ep) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in keys + ["seed"]]
    rows = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row["n"] = len(g)
        for m in metrics:
            row[m] = iqm(g[m].dropna())
            row[f"{m}_lo"], row[f"{m}_hi"] = boot_ci(g[m].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def errline(ax, x, s, col, color, label):
    ax.errorbar(x, s[col], yerr=[s[col] - s[f"{col}_lo"], s[f"{col}_hi"] - s[col]], color=color, marker="o",
                markersize=6, capsize=3, label=label, markeredgecolor=SURFACE, linewidth=2)


def fig_adv_scale(summ: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    mults = sorted(summ.reward_multiplier.unique())
    x = np.arange(len(mults))
    for color, na in zip([CAT["variance"], CAT["bias²"]], [True, False]):
        s = summ[summ.norm_adv == na].set_index("reward_multiplier").reindex(mults)
        lab = "adv. normalisation ON" if na else "adv. normalisation OFF"
        for ax, col in zip(axes, ["adv_mean_abs", "kl_p95", "spike_frac", "final_return"]):
            errline(ax, x, s, col, color, lab)
    titles = ["Raw advantage scale", "Update size (95th pct KL)", "Updates with KL > 0.05", "Final return"]
    ylabels = ["median |A| per update (log)", "KL(π_old‖π_new), log", "fraction of updates", "episode return (IQM, 95% CI)"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_xticks(x, [f"×{m:g}" for m in mults])
        ax.set_xlabel("reward multiplier")
        ax.set_title(t, loc="left", fontsize=10)
        ax.set_ylabel(yl)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Intervention A: scaling the reward (Hopper, return-scaling off, 5 seeds)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p4_intervention_adv_scale.png")


def fig_safety(summ: pd.DataFrame, per_seed: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    w = sorted(summ.cost_weight.unique())
    x = np.arange(len(w))
    s = summ.set_index("cost_weight").reindex(w)
    for ax, col, t, yl in zip(
        axes[:3],
        ["cost_rate", "adv_kurtosis", "kl_p95"],
        ["Constraint violations", "Advantage tail heaviness", "Update size (95th pct KL)"],
        ["fraction of steps over speed limit", "median kurtosis of A per update", "KL(π_old‖π_new); no update exceeded 0.05"],
    ):
        errline(ax, x, s, col, CAT["variance"], None)
        ax.set_xticks(x, [f"{v:g}" for v in w])
        ax.set_xlabel("penalty weight w")
        ax.set_title(t, loc="left", fontsize=10)
        ax.set_ylabel(yl)
    ax = axes[3]
    ramp = dict(zip(w, LAMBDA_RAMP))
    for wv in w:
        g = per_seed[per_seed.cost_weight == wv]
        ax.scatter(g.cost_rate, g.final_return, s=40, color=ramp[wv], edgecolor=SURFACE, linewidth=1.2, label=f"w = {wv:g}", zorder=3)
    ax.set_xlabel("constraint violation rate (last 10%)")
    ax.set_ylabel("task return (last 10%)")
    ax.set_title("Return–safety trade-off (one dot per seed)", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Intervention B: safety-penalty weight  ·  Hopper velocity constraint, normalised PPO, 5 seeds",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p4_intervention_safety_penalty.png")


def main():
    os.makedirs(TAB, exist_ok=True)
    runs = load_sweep("lambda_sweep")
    c = correlations(runs)
    c.to_csv(os.path.join(TAB, "p4_correlations_per_run.csv"), index=False)
    summ = c.groupby(["env_id", "mode", "adv", "inst"]).rho.agg(["median", "count", lambda r: (r > 0).mean()])
    summ.columns = ["rho_median", "n_runs", "frac_positive"]
    summ.reset_index().to_csv(os.path.join(TAB, "p4_correlation_summary.csv"), index=False)
    print(summ.loc[(slice(None), "detrended"), :].round(3).to_string())
    fig_corr_heatmap(c)
    fig_binned(runs)

    a = load_sweep("adv_scale")
    if a:
        per = run_summary(a, ["norm_adv", "reward_multiplier"])
        s = agg(per, ["norm_adv", "reward_multiplier"])
        s.to_csv(os.path.join(TAB, "p4_adv_scale_summary.csv"), index=False)
        print(s[["norm_adv", "reward_multiplier", "n", "adv_mean_abs", "kl_median", "kl_p95", "spike_frac",
                 "grad_norm", "final_return", "final_return_lo", "final_return_hi"]].round(4).to_string(index=False))
        fig_adv_scale(s)

    b = load_sweep("safety_penalty")
    if b:
        per = run_summary(b, ["cost_weight"])
        s = agg(per, ["cost_weight"])
        s.to_csv(os.path.join(TAB, "p4_safety_penalty_summary.csv"), index=False)
        per.to_csv(os.path.join(TAB, "p4_safety_penalty_per_seed.csv"), index=False)
        print(s[["cost_weight", "n", "final_return", "cost_rate", "adv_kurtosis", "kl_median", "kl_p95", "spike_frac"]]
              .round(4).to_string(index=False))
        fig_safety(s, per)


if __name__ == "__main__":
    main()
