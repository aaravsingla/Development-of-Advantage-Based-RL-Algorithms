"""PPO with clipped surrogate objective (Schulman et al., 2017), from scratch.

Design choices follow the well-tested "37 implementation details" recipe (Huang et al., 2022)
so that differences between runs come from the variable under study (GAE lambda, reward
noise, ...) and not from a weak baseline. Every recipe component is a config toggle.

Outputs per run directory:
    config.json      full config
    episodes.csv     global_step, clean episodic return, length    (learning curves)
    updates.csv      one row per PPO update: losses, KL, clip-frac, raw advantage stats, ...
    ckpt_<step>.pt   actor, critic, obs-normaliser and reward-scaler state
    done             marker written when training finishes
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis

from envs.wrappers import VecEnv
from rl.gae import compute_gae
from rl.networks import Critic, GaussianActor
from rl.normalize import ObsNormalizer, RewardScaler


@dataclass
class PPOConfig:
    env_id: str = "Hopper-v5"
    seed: int = 0
    total_steps: int = 1_000_000
    num_envs: int = 1
    num_steps: int = 2048  # rollout length per env; batch = num_envs * num_steps
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 32
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    norm_obs: bool = True
    scale_reward: bool = True
    reward_noise: float = 0.0  # std of Gaussian noise added to the training reward
    reward_multiplier: float = 1.0  # r -> c*r (Phase 4: changes advantage magnitude)
    cost_type: str = ""  # "velocity": Safety-Gymnasium-style speed limit (Phase 4)
    cost_weight: float = 0.0  # train on r - w*c
    cost_threshold: float = 0.7402
    env_kwargs: dict | None = None  # passed to gym.make (Phase 5: SafeNav reward mode / coef)
    # PPO-Lagrangian (Phase 5): train on r - lam * c, lam <- max(0, lam + lam_lr * (Jc - cost_limit))
    # where Jc is the mean episode cost of the latest rollout. Single combined critic (penalty form).
    lagrangian: bool = False
    cost_limit: float = 0.05
    lam_lr: float = 0.05
    lam_init: float = 0.0
    hidden: int = 64
    checkpoint_every: int = 100_000
    out_dir: str = "results/runs/debug"


class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def save_checkpoint(path, actor, critic, obs_norm, reward_scaler, cfg, global_step):
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_rms": obs_norm.rms.state_dict() if obs_norm else None,
            "reward_rms": reward_scaler.rms.state_dict() if reward_scaler else None,
            "config": asdict(cfg),
            "global_step": global_step,
        },
        path,
    )


def advantage_stats(adv: np.ndarray) -> dict:
    a = np.abs(adv)
    return {
        "adv_mean_abs": float(a.mean()),
        "adv_std": float(adv.std()),
        "adv_p99_abs": float(np.percentile(a, 99)),
        "adv_max_abs": float(a.max()),
        "adv_kurtosis": float(kurtosis(adv)),
    }


def train(cfg: PPOConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = VecEnv(cfg.env_id, cfg.num_envs, cfg.seed, cfg.reward_noise, reward_multiplier=cfg.reward_multiplier,
                 cost_type=cfg.cost_type, cost_weight=cfg.cost_weight, cost_threshold=cfg.cost_threshold,
                 env_kwargs=cfg.env_kwargs)
    lam = cfg.lam_init
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    actor = GaussianActor(obs_dim, act_dim, cfg.hidden)
    critic = Critic(obs_dim, cfg.hidden)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)

    obs_norm = ObsNormalizer((obs_dim,)) if cfg.norm_obs else None
    reward_scaler = RewardScaler(cfg.num_envs, cfg.gamma) if cfg.scale_reward else None
    norm = (lambda o, update=True: obs_norm(o, update)) if obs_norm else (lambda o, update=True: o.astype(np.float32))

    T, N = cfg.num_steps, cfg.num_envs
    batch_size = T * N
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_steps // batch_size

    buf_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
    buf_act = np.zeros((T, N, act_dim), dtype=np.float32)
    buf_logp = np.zeros((T, N), dtype=np.float32)
    buf_rew = np.zeros((T, N))
    buf_val = np.zeros((T, N))
    buf_next_val = np.zeros((T, N))
    buf_term = np.zeros((T, N))
    buf_end = np.zeros((T, N))

    ep_log = CsvLogger(os.path.join(cfg.out_dir, "episodes.csv"))
    upd_log = CsvLogger(os.path.join(cfg.out_dir, "updates.csv"))

    obs = norm(env.reset())
    global_step = 0
    next_ckpt = cfg.checkpoint_every
    start = time.time()

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = cfg.lr * (1.0 - (update - 1.0) / num_updates)

        # ---------------- rollout ----------------
        trunc_values = {}  # (t, env) -> V(true final obs) for time-limit truncations
        rollout_costs = []
        for t in range(T):
            with torch.no_grad():
                o = torch.as_tensor(obs)
                dist = actor.dist(o)
                action = dist.sample()
                buf_logp[t] = dist.log_prob(action).sum(-1).numpy()
                buf_val[t] = critic(o).numpy()
            buf_obs[t] = obs
            buf_act[t] = action.numpy()

            next_obs_raw, rew, term, trunc, true_next_raw, finished = env.step(action.numpy())
            global_step += N
            end = np.maximum(term, trunc)
            if cfg.lagrangian:
                rew = rew - lam * env.last_costs
            buf_rew[t] = reward_scaler(rew, end) if reward_scaler else rew
            buf_term[t] = term
            buf_end[t] = end

            for i in np.flatnonzero(trunc * (1 - term)):
                with torch.no_grad():
                    v = critic(torch.as_tensor(norm(true_next_raw[i : i + 1], update=False))).item()
                trunc_values[(t, i)] = v
            for i, ret, length, cost, success in finished:
                ep_log.log({"global_step": global_step, "return": ret, "length": length, "cost": cost, "success": success})
                rollout_costs.append(cost)

            obs = norm(next_obs_raw)

        if cfg.lagrangian and rollout_costs:
            lam = max(0.0, lam + cfg.lam_lr * (float(np.mean(rollout_costs)) - cfg.cost_limit))

        # ---------------- GAE ----------------
        with torch.no_grad():
            last_val = critic(torch.as_tensor(obs)).numpy()
        buf_next_val[:-1] = buf_val[1:]
        buf_next_val[-1] = last_val
        for (t, i), v in trunc_values.items():
            buf_next_val[t, i] = v
        # A rollout cut at t = T-1 is handled automatically: episode_end is 0 there, so the
        # recursion simply starts from A_T = 0 and delta_{T-1} bootstraps from V(s_T).
        adv, ret = compute_gae(buf_rew, buf_val, buf_next_val, buf_term, buf_end, cfg.gamma, cfg.gae_lambda)

        # ---------------- PPO update ----------------
        b_obs = torch.as_tensor(buf_obs.reshape(batch_size, obs_dim))
        b_act = torch.as_tensor(buf_act.reshape(batch_size, act_dim))
        b_logp = torch.as_tensor(buf_logp.reshape(batch_size))
        b_adv = torch.as_tensor(adv.reshape(batch_size), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(batch_size), dtype=torch.float32)
        b_val = torch.as_tensor(buf_val.reshape(batch_size), dtype=torch.float32)

        clipfracs, kls, grad_norms = [], [], []
        pg_losses, v_losses, entropies = [], [], []
        for epoch in range(cfg.update_epochs):
            perm = np.random.permutation(batch_size)
            for s in range(0, batch_size, minibatch_size):
                mb = perm[s : s + minibatch_size]
                dist = actor.dist(b_obs[mb])
                new_logp = dist.log_prob(b_act[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean().item())
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                ).mean()

                new_v = critic(b_obs[mb])
                if cfg.clip_vloss:
                    v_clipped = b_val[mb] + torch.clamp(new_v - b_val[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max((new_v - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - b_ret[mb]) ** 2).mean()

                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                grad_norms.append(nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item())
                optimizer.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

        # KL between the pre-update and post-update policy on the whole batch:
        # the cleanest single number for "how big was this policy update".
        with torch.no_grad():
            log_ratio = actor.dist(b_obs).log_prob(b_act).sum(-1) - b_logp
            kl_full = ((log_ratio.exp() - 1) - log_ratio).mean().item()
            y_var = b_ret.var().item()
            explained_var = 1 - (b_ret - b_val).var().item() / y_var if y_var > 0 else float("nan")

        row = {
            "update": update,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "pg_loss": np.mean(pg_losses),
            "v_loss": np.mean(v_losses),
            "entropy": np.mean(entropies),
            "approx_kl_mb": np.mean(kls),
            "kl_full": kl_full,
            "clipfrac": np.mean(clipfracs),
            "grad_norm": np.mean(grad_norms),
            "grad_norm_max": np.max(grad_norms),
            "explained_var": explained_var,
            "value_mean": float(buf_val.mean()),
            "return_target_mean": float(ret.mean()),
            "reward_scale": reward_scaler.scale if reward_scaler else 1.0,
            "log_std_mean": actor.log_std.mean().item(),
            "lagrange_lambda": lam,
            "sps": int(global_step / (time.time() - start)),
        }
        row.update(advantage_stats(adv.reshape(-1)))
        upd_log.log(row)

        if global_step >= next_ckpt or update == num_updates:
            save_checkpoint(
                os.path.join(cfg.out_dir, f"ckpt_{global_step}.pt"), actor, critic, obs_norm, reward_scaler, cfg, global_step
            )
            next_ckpt += cfg.checkpoint_every

    env.close()
    ep_log.close()
    upd_log.close()
    open(os.path.join(cfg.out_dir, "done"), "w").close()
    return cfg.out_dir
