"""Regenerates every Phase-2 figure and table from raw run logs and probe outputs.

    python -m analysis.phase2

Aggregation across seeds uses the interquartile mean (IQM; Agarwal et al., 2021) with
95% bootstrap confidence intervals over seeds. All inputs are optional: whatever exists
under results/ is plotted, so the script can be run while sweeps are still in progress.
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
from scipy.stats import trim_mean

RUNS = "results/runs"
PROBES = "results/probes"
FIG = "results/figures"
TAB = "results/tables"

# ---- palette (validated: ordinal blue ramp for lambda, categorical slots for bias/var/mse) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAMBDA_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CAT = {"variance": "#2a78d6", "bias²": "#eb6834", "MSE": "#1baf7a"}
CURVE_LAMBDAS = [0.0, 0.8, 0.95, 1.0]
NOISE_LAMBDAS = [0.5, 0.9, 0.95, 1.0]
BIN = 25_000

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2, "lines.linewidth": 2,
    "font.family": "sans-serif", "figure.dpi": 130,
})


def iqm(x) -> float:
    return float(trim_mean(np.asarray(x), 0.25)) if len(x) else np.nan


def boot_ci(x, stat=iqm, n=2000, seed=0) -> tuple[float, float]:
    x = np.asarray(x)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return tuple(np.percentile(s, [2.5, 97.5]))


def short_env(env_id: str) -> str:
    return env_id.replace("Continuous", "").split("-")[0]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG, name))


# ----------------------------------------------------------------------------- run loading
def load_runs(sweeps: list[str]) -> pd.DataFrame:
    rows = []
    for sweep in sweeps:
        for d in sorted(glob.glob(os.path.join(RUNS, sweep, "*"))):
            if not os.path.exists(os.path.join(d, "done")):
                continue
            cfg = json.load(open(os.path.join(d, "config.json")))
            rows.append({"dir": d, "env_id": cfg["env_id"], "lam": cfg["gae_lambda"], "seed": cfg["seed"],
                         "sigma": cfg["reward_noise"], "total_steps": cfg["total_steps"]})
    return pd.DataFrame(rows)


def binned_curve(run_dir: str, total_steps: int) -> np.ndarray:
    ep = pd.read_csv(os.path.join(run_dir, "episodes.csv"))
    edges = np.arange(0, total_steps + BIN, BIN)
    idx = np.digitize(ep.global_step, edges) - 1
    means = pd.Series(ep["return"].values).groupby(idx).mean()
    curve = np.full(len(edges) - 1, np.nan)
    for i, v in means.items():
        if 0 <= i < len(curve):
            curve[i] = v
    return pd.Series(curve).ffill().bfill().values


def run_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run scalar metrics: final return, AUC, worst drawdown (normalised per env)."""
    out = []
    for _, r in runs.iterrows():
        c = binned_curve(r.dir, r.total_steps)
        ep = pd.read_csv(os.path.join(r.dir, "episodes.csv"))
        final = ep[ep.global_step > 0.9 * r.total_steps]["return"].mean()
        out.append({**r.to_dict(), "final": final, "auc": np.nanmean(c), "curve": c})
    m = pd.DataFrame(out)
    # drawdown: largest fall from the running best, as a fraction of the env's return range
    for env, g in m.groupby("env_id"):
        allc = np.concatenate(g.curve.values)
        rng = np.nanmax(allc) - np.nanmin(allc)
        for i in g.index:
            c = m.at[i, "curve"]
            m.at[i, "drawdown"] = float(np.max(np.maximum.accumulate(c) - c) / rng)
    return m


def summarise(m: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in m.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row["n_seeds"] = len(g)
        for col in ["final", "auc", "drawdown"]:
            row[f"{col}_iqm"] = iqm(g[col])
            row[f"{col}_lo"], row[f"{col}_hi"] = boot_ci(g[col])
        row["final_seed_std"] = g["final"].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_learning_curves(m: pd.DataFrame):
    envs = sorted(m.env_id.unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 3.6), squeeze=False)
    for ax, env in zip(axes[0], envs):
        g = m[(m.env_id == env) & (m.sigma == 0)]
        for color, lam in zip(LAMBDA_RAMP, CURVE_LAMBDAS):
            curves = np.stack(g[g.lam == lam].curve.values) if (g.lam == lam).any() else None
            if curves is None:
                continue
            x = (np.arange(curves.shape[1]) + 1) * BIN / 1e6
            mid = np.array([iqm(curves[:, t]) for t in range(curves.shape[1])])
            ci = np.array([boot_ci(curves[:, t], n=300) for t in range(curves.shape[1])])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mid, color=color, label=f"λ = {lam:g}")
            ax.annotate(f"λ={lam:g}", (x[-1], mid[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        ax.set_title(short_env(env), loc="left")
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("episode return (IQM, 95% CI)")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("PPO learning curves by GAE λ  ·  5 seeds", x=0.01, ha="left", color=INK, fontsize=12)
    save(fig, "p2_learning_curves.png")


def fig_lambda_summary(summ: pd.DataFrame):
    envs = sorted(summ.env_id.unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 3.3), squeeze=False)
    for ax, env in zip(axes[0], envs):
        s = summ[(summ.env_id == env) & (summ.sigma == 0)].sort_values("lam")
        x = np.arange(len(s))
        ax.errorbar(x, s.final_iqm, yerr=[s.final_iqm - s.final_lo, s.final_hi - s.final_iqm], color=CAT["variance"],
                    marker="o", markersize=6, capsize=3, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5)
        best = s.final_iqm.values.argmax()
        ax.annotate(f"{s.final_iqm.values[best]:.0f}", (x[best], s.final_iqm.values[best]), xytext=(0, 9),
                    textcoords="offset points", ha="center", color=INK, fontsize=8.5)
        ax.set_xticks(x, [f"{v:g}" for v in s.lam])
        ax.set_title(short_env(env), loc="left")
        ax.set_xlabel("GAE λ")
        ax.set_ylabel("final return (last 10%, IQM, 95% CI)")
    fig.suptitle("Final performance vs λ", x=0.01, ha="left", color=INK, fontsize=12)
    save(fig, "p2_lambda_final_return.png")


def fig_grad_probe(df: pd.DataFrame):
    """rows = env, cols = training stage. Lines: relative variance, bias², MSE vs lambda (log scale)."""
    df = df[df.sigma == 0]
    envs = sorted(df.env_id.unique())
    stages = ["early", "mid", "late"]
    fig, axes = plt.subplots(len(envs), 3, figsize=(13, 3.3 * len(envs)), squeeze=False, sharex=True)
    lam_vals = sorted(df.lam.unique())
    x = np.arange(len(lam_vals))
    for i, env in enumerate(envs):
        for j, stage in enumerate(stages):
            ax = axes[i][j]
            g = df[(df.env_id == env) & (df.stage == stage)]
            if g.empty:
                ax.set_visible(False)
                continue
            for name, col in [("variance", "var_ratio"), ("bias²", "bias2_ratio"), ("MSE", "mse_ratio")]:
                agg = g.groupby("lam")[col].agg(["mean", "min", "max"]).reindex(lam_vals)
                agg = agg.where(agg > 0)  # bias^2 below measurement resolution is left as a gap
                y = agg["mean"]
                ax.fill_between(x, agg["min"], agg["max"], color=CAT[name], alpha=0.15, linewidth=0)
                ax.plot(x, y, color=CAT[name], marker="o", markersize=5, label=name,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
            mse = g.groupby("lam")["mse_ratio"].mean().reindex(lam_vals)
            k = int(np.nanargmin(mse.values))
            ax.axvline(x[k], color=AXIS, linewidth=1, linestyle="--", zorder=0)
            ax.annotate(f"min MSE at λ={lam_vals[k]:g}", (x[k], mse.values[k]),
                        xytext=(6, -14), textcoords="offset points", color=INK2, fontsize=8)
            ax.set_yscale("log")
            step = g.global_step.mean() / 1e6
            ax.set_title(f"{short_env(env)} · {stage} ({step:.2f}M steps)", loc="left", fontsize=10)
            ax.set_xticks(x, [f"{v:g}" for v in lam_vals])
            if j == 0:
                ax.set_ylabel("÷ Monte Carlo (λ=1) variance  (log)")
            if i == len(envs) - 1:
                ax.set_xlabel("GAE λ")
    axes[0][0].legend(loc="lower left")
    fig.suptitle("Policy-gradient bias² vs variance, measured on frozen policies "
                 "(line = mean over probed seeds, band = range; gaps = bias² below resolution)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "p2_gradient_bias_variance.png")



def fig_noise(m_noise: pd.DataFrame, adv_noise: pd.DataFrame | None, probe_noise: pd.DataFrame | None):
    if not m_noise.empty:
        summ = summarise(m_noise, ["lam", "sigma"])
        summ.to_csv(os.path.join(TAB, "p2_noise_summary.csv"), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
        sig = sorted(summ.sigma.unique())
        x = np.arange(len(sig))
        for color, lam in zip(LAMBDA_RAMP, NOISE_LAMBDAS):
            s = summ[summ.lam == lam].set_index("sigma").reindex(sig)
            axes[0].errorbar(x, s.final_iqm, yerr=[s.final_iqm - s.final_lo, s.final_hi - s.final_iqm], color=color,
                             marker="o", markersize=6, capsize=3, label=f"λ = {lam:g}", markeredgecolor=SURFACE)
            axes[1].plot(x, s.drawdown_iqm, color=color, marker="o", markersize=6, label=f"λ = {lam:g}", markeredgecolor=SURFACE)
        for ax in axes:
            ax.set_xticks(x, [f"{v:g}" for v in sig])
            ax.set_xlabel("reward-noise std σ")
        axes[0].set_ylabel("final clean return (IQM, 95% CI)")
        axes[1].set_ylabel("worst drawdown (fraction of return range)")
        axes[0].set_title("Performance under noisy rewards", loc="left")
        axes[1].set_title("Learning-curve instability", loc="left")
        axes[0].legend()
        fig.suptitle("LunarLander: GAE λ vs reward noise  ·  5 seeds", x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
        save(fig, "p2_noise_learning.png")

    if adv_noise is not None and not adv_noise.empty:
        envs = sorted(probe_noise.env_id.unique()) if probe_noise is not None and not probe_noise.empty else []
        fig, axes = plt.subplots(1, 1 + len(envs), figsize=(4.6 * (1 + len(envs)), 3.6), squeeze=False)
        ax = axes[0][0]
        ax.scatter(adv_noise.noise_var_theory, adv_noise.noise_var_measured, s=30, color=CAT["variance"],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        lo = adv_noise[["noise_var_theory", "noise_var_measured"]].min().min()
        hi = adv_noise[["noise_var_theory", "noise_var_measured"]].max().max()
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, linestyle="--", zorder=1)
        ax.annotate("y = x", (hi, hi), xytext=(-30, -4), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("predicted  (σ/scale)² · Σ(γλ)^{2l}")
        ax.set_ylabel("measured variance of noise in Â")
        ax.set_title("Advantage noise: theory vs measured", loc="left")
        for ax, env in zip(axes[0][1:], envs):
            late = probe_noise[(probe_noise.stage == "late") & (probe_noise.env_id == env)]
            sig = sorted(late.sigma.unique())
            x = np.arange(len(sig))
            for color, lam in zip(LAMBDA_RAMP, NOISE_LAMBDAS):
                g = late[late.lam == lam].groupby("sigma")["var_ratio"].mean().reindex(sig)
                ax.plot(x, g.values, color=color, marker="o", markersize=5, markeredgecolor=SURFACE, label=f"λ = {lam:g}")
            ax.set_yscale("log")
            ax.set_xticks(x, [f"{v:g}" for v in sig])
            ax.set_xlabel("reward-noise std σ")
            ax.set_ylabel("gradient variance ÷ clean Monte Carlo")
            ax.set_title(f"{short_env(env)} · gradient variance (late ckpt)", loc="left")
            ax.legend(fontsize=7.5)
        fig.tight_layout()
        save(fig, "p2_noise_theory.png")


def fig_value_bias(v: pd.DataFrame):
    envs = sorted(v.env_id.unique())
    fig, axes = plt.subplots(2, len(envs), figsize=(5.2 * len(envs), 6.6), squeeze=False)
    for j, env in enumerate(envs):
        g = v[(v.env_id == env) & (v.sigma == 0)]
        ax = axes[0][j]
        for color, lam in zip(LAMBDA_RAMP, CURVE_LAMBDAS):
            h = g[g.lam == lam].groupby("global_step")["rel_bias"].agg(list)
            if h.empty:
                continue
            mid = [iqm(x) for x in h.values]
            ci = np.array([boot_ci(x, n=300) for x in h.values])
            ax.fill_between(h.index / 1e6, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
            ax.plot(h.index / 1e6, mid, color=color, label=f"λ = {lam:g}")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(f"{short_env(env)} · critic bias over training", loc="left")
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("(V − G) / mean|G|   (>0 = overestimate)")
        ax.legend(fontsize=7.5)

        ax = axes[1][j]
        late = g[g.global_step >= 0.7 * g.global_step.max()].groupby(["lam", "seed"])[["rel_bias", "rel_rmse"]].mean().reset_index()
        lam_vals = sorted(late.lam.unique())
        x = np.arange(len(lam_vals))
        for name, col, color in [("bias", "rel_bias", CAT["bias²"]), ("RMSE", "rel_rmse", CAT["variance"])]:
            mids = [iqm(late[late.lam == l][col]) for l in lam_vals]
            ci = np.array([boot_ci(late[late.lam == l][col]) for l in lam_vals])
            ax.errorbar(x, mids, yerr=[np.array(mids) - ci[:, 0], ci[:, 1] - np.array(mids)], color=color, marker="o",
                        markersize=5, capsize=3, label=name, markeredgecolor=SURFACE)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(x, [f"{l:g}" for l in lam_vals])
        ax.set_xlabel("GAE λ (critic trained with TD(λ) targets)")
        ax.set_ylabel("relative to mean|G|, last 30% of training")
        ax.set_title(f"{short_env(env)} · critic error vs λ", loc="left")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "p2_value_bias.png")


def main():
    os.makedirs(TAB, exist_ok=True)

    runs = load_runs(["lambda_sweep"])
    if not runs.empty:
        m = run_metrics(runs)
        summ = summarise(m, ["env_id", "lam", "sigma"])
        summ.to_csv(os.path.join(TAB, "p2_lambda_summary.csv"), index=False)
        print(summ[["env_id", "lam", "n_seeds", "final_iqm", "final_lo", "final_hi", "auc_iqm", "drawdown_iqm"]].round(3).to_string(index=False))
        fig_learning_curves(m)
        fig_lambda_summary(summ)
    else:
        m = pd.DataFrame()

    grad_files = glob.glob(os.path.join(PROBES, "grad_*", "grad_stats.csv"))
    probe = None
    if grad_files:
        parts = []
        for f in grad_files:
            d = pd.read_csv(f)
            meta = json.load(open(os.path.join(os.path.dirname(f), "meta.json")))
            d["env_id"], d["probe_seed"] = meta["env_id"], meta["seed"]
            order = sorted(d.global_step.unique())
            d["stage"] = d.global_step.map(dict(zip(order, ["early", "mid", "late"][: len(order)])))
            parts.append(d)
        probe = pd.concat(parts)
        probe.to_csv(os.path.join(TAB, "p2_gradient_probe_all.csv"), index=False)
        summary = (probe[probe.sigma == 0].groupby(["env_id", "stage", "lam"])[["var_ratio", "bias2_ratio", "mse_ratio", "snr_mc", "mean_batch_size"]]
                   .mean().reset_index())
        summary.to_csv(os.path.join(TAB, "p2_gradient_probe_summary.csv"), index=False)
        print(summary.round(4).to_string(index=False))
        fig_grad_probe(probe)

    adv_files = glob.glob(os.path.join(PROBES, "grad_*", "adv_noise_stats.csv"))
    adv_noise = pd.concat([pd.read_csv(f) for f in adv_files]) if adv_files else None
    noise_runs = load_runs(["noise_sweep"])
    m_noise = pd.DataFrame()
    if not noise_runs.empty:
        base = runs[(runs.env_id == "LunarLanderContinuous-v3") & runs.lam.isin(NOISE_LAMBDAS)] if not runs.empty else pd.DataFrame()
        m_noise = run_metrics(pd.concat([base, noise_runs]))
    fig_noise(m_noise, adv_noise, probe[probe.sigma.isin([0, 1, 3, 10])] if probe is not None else None)

    vf = os.path.join(PROBES, "values_lambda_sweep.csv")
    if os.path.exists(vf):
        v = pd.read_csv(vf)
        fig_value_bias(v)
        late = v[v.global_step >= 0.7 * v.groupby("env_id").global_step.transform("max")]
        late.groupby(["env_id", "lam"])[["rel_bias", "rel_rmse", "frac_over"]].mean().reset_index().to_csv(
            os.path.join(TAB, "p2_value_bias_summary.csv"), index=False)


if __name__ == "__main__":
    main()
