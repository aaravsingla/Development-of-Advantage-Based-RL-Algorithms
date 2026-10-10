"""DDPG (Lillicrap et al., 2016) and TD3 (Fujimoto et al., 2018), from scratch, in one class.

TD3 is DDPG plus three changes, each a separate flag so they can be ablated:
    twin_q          clipped double-Q: target uses min(Q1', Q2')     -> counters overestimation
    target_noise    target policy smoothing: a' = mu'(s') + clip(N(0, 0.2), -0.5, 0.5)
    policy_delay    update actor/targets every 2 critic updates
With all three off this is the "Our DDPG" baseline from the TD3 paper (same nets, same lr),
so any difference in value-estimation bias is caused by those three changes only.

Why DDPG overestimates: the actor is trained to maximise Q, so it seeks out actions where
the critic's error is positive; the bootstrapped target then propagates that positive error
(E[max] >= max E). Clipped double-Q takes the min of two independently-trained critics,
which trades this for a mild underestimation.

Outputs per run directory:
    config.json, episodes.csv, updates.csv
    value_est.csv   every `eval_every` steps: Q(s, mu(s)) vs realised discounted return on
                    deterministic evaluation episodes (the Fujimoto et al. Fig. 1 measurement)
    ckpt_<step>.pt, done
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.ppo import CsvLogger


@dataclass
class DDPGConfig:
    env_id: str = "Hopper-v5"
    algo: str = "ddpg"  # "ddpg" or "td3" (sets the three flags below)
    seed: int = 0
    total_steps: int = 1_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden: int = 256
    learning_starts: int = 25_000  # uniform-random actions before this
    expl_noise: float = 0.1  # Gaussian exploration noise, fraction of action range
    twin_q: bool | None = None
    target_noise: float | None = None
    target_noise_clip: float = 0.5
    policy_delay: int | None = None
    eval_every: int = 50_000
    eval_episodes: int = 10
    checkpoint_every: int = 250_000
    out_dir: str = "results/runs/debug_ddpg"

    def __post_init__(self):
        td3 = self.algo == "td3"
        if self.twin_q is None:
            self.twin_q = td3
        if self.target_noise is None:
            self.target_noise = 0.2 if td3 else 0.0
        if self.policy_delay is None:
            self.policy_delay = 2 if td3 else 1


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, act_scale, act_bias):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, act_dim), nn.Tanh())
        self.register_buffer("scale", torch.as_tensor(act_scale, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor(act_bias, dtype=torch.float32))

    def forward(self, obs):
        return self.net(obs) * self.scale + self.bias


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros(size, np.float32)
        self.term = np.zeros(size, np.float32)  # only true terminals; truncation still bootstraps
        self.size, self.ptr, self.max = 0, 0, size

    def add(self, o, a, r, o2, term):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.term[i] = o, a, r, o2, term
        self.ptr = (i + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, n)
        t = lambda x: torch.as_tensor(x[idx])
        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), t(self.term)


def soft_update(target: nn.Module, src: nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), src.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)


def discounted_returns(r: np.ndarray, gamma: float) -> np.ndarray:
    g, acc = np.zeros(len(r)), 0.0
    for t in reversed(range(len(r))):
        acc = r[t] + gamma * acc
        g[t] = acc
    return g


def evaluate_values(env, actor, critics, gamma, n_episodes, seed) -> dict:
    """Deterministic episodes; compare Q(s_t, mu(s_t)) with the realised discounted return G_t.

    States within H steps of a time-limit truncation are dropped (missing tail < 1% of G).
    """
    H = int(np.ceil(np.log(0.01) / np.log(gamma)))
    q_all, g_all, returns, q0, g0 = [], [], [], [], []
    for k in range(n_episodes):
        o, _ = env.reset(seed=seed + k)
        obs, rews, done, term = [], [], False, False
        while not done:
            with torch.no_grad():
                a = actor(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()
            obs.append(o)
            o, r, term, trunc, _ = env.step(a)
            rews.append(r)
            done = term or trunc
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32)
        with torch.no_grad():
            acts = actor(obs)
            q = critics[0](obs, acts).numpy()  # Q1 is what the actor maximises
        g = discounted_returns(np.array(rews), gamma)
        keep = np.ones(len(g), bool) if term else np.arange(len(g)) < len(g) - H
        q_all.append(q[keep])
        g_all.append(g[keep])
        returns.append(float(np.sum(rews)))
        if keep[0]:
            q0.append(q[0])
            g0.append(g[0])
    q, g = np.concatenate(q_all), np.concatenate(g_all)
    if len(g) == 0:  # every episode was time-limit truncated and shorter than H: nothing valid to score
        return {"eval_return": float(np.mean(returns)), "n_states": 0}
    e = q - g
    return {
        "eval_return": float(np.mean(returns)),
        "n_states": len(e),
        "q_mean": float(q.mean()),
        "g_mean": float(g.mean()),
        "bias": float(e.mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "err_std": float(e.std()),
        "mean_abs_return": float(np.abs(g).mean()),
        "rel_bias": float(e.mean() / np.abs(g).mean()),
        "rel_rmse": float(np.sqrt((e**2).mean()) / np.abs(g).mean()),
        "rel_err_std": float(e.std() / np.abs(g).mean()),
        "frac_over": float((e > 0).mean()),
        "q0_minus_g0": float(np.mean(np.array(q0) - np.array(g0))) if q0 else float("nan"),
    }


def train(cfg: DDPGConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env, eval_env = gym.make(cfg.env_id), gym.make(cfg.env_id)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    act_scale, act_bias = (high - low) / 2, (high + low) / 2
    env.action_space.seed(cfg.seed)

    actor = Actor(obs_dim, act_dim, cfg.hidden, act_scale, act_bias)
    critics = nn.ModuleList([QNet(obs_dim, act_dim, cfg.hidden) for _ in range(2 if cfg.twin_q else 1)])
    actor_t, critics_t = copy.deepcopy(actor), copy.deepcopy(critics)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critics.parameters(), lr=cfg.critic_lr)
    buf = ReplayBuffer(obs_dim, act_dim, min(cfg.buffer_size, cfg.total_steps))
    scale_t = torch.as_tensor(act_scale, dtype=torch.float32)
    low_t, high_t = torch.as_tensor(low), torch.as_tensor(high)

    ep_log = CsvLogger(os.path.join(cfg.out_dir, "episodes.csv"))
    upd_log = CsvLogger(os.path.join(cfg.out_dir, "updates.csv"))
    val_log = CsvLogger(os.path.join(cfg.out_dir, "value_est.csv"))

    o, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_len, n_updates = 0.0, 0, 0
    stats = {"critic_loss": [], "actor_loss": [], "q_pred": [], "q_target": []}
    start = time.time()

    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.learning_starts:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()
            a = np.clip(a + rng.normal(0, cfg.expl_noise * act_scale), low, high)
        o2, r, term, trunc, _ = env.step(a)
        buf.add(o, a, r, o2, float(term))
        o = o2
        ep_ret += r
        ep_len += 1
        if term or trunc:
            ep_log.log({"global_step": step, "return": ep_ret, "length": ep_len})
            o, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step > cfg.learning_starts:
            ob, ac, rw, ob2, tm = buf.sample(cfg.batch_size, rng)
            with torch.no_grad():
                a2 = actor_t(ob2)
                if cfg.target_noise > 0:
                    noise = (torch.randn_like(a2) * cfg.target_noise).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
                    a2 = torch.max(torch.min(a2 + noise * scale_t, high_t), low_t)
                q_next = torch.stack([c(ob2, a2) for c in critics_t]).min(0).values
                y = rw + cfg.gamma * (1 - tm) * q_next
            qs = [c(ob, ac) for c in critics]
            critic_loss = sum(F.mse_loss(q, y) for q in qs)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            n_updates += 1
            stats["critic_loss"].append(critic_loss.item())
            stats["q_pred"].append(qs[0].mean().item())
            stats["q_target"].append(y.mean().item())

            if n_updates % cfg.policy_delay == 0:
                actor_loss = -critics[0](ob, actor(ob)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                soft_update(actor_t, actor, cfg.tau)
                soft_update(critics_t, critics, cfg.tau)
                stats["actor_loss"].append(actor_loss.item())

        if step % 5000 == 0 and stats["critic_loss"]:
            upd_log.log({"global_step": step, **{k: float(np.mean(v)) for k, v in stats.items() if v},
                         "sps": int(step / (time.time() - start))})
            stats = {k: [] for k in stats}

        if step % cfg.eval_every == 0:
            val_log.log({"global_step": step, **evaluate_values(eval_env, actor, critics, cfg.gamma,
                                                                cfg.eval_episodes, seed=10_000 + step)})
        if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
            torch.save({"actor": actor.state_dict(), "critics": critics.state_dict(), "config": asdict(cfg),
                        "global_step": step}, os.path.join(cfg.out_dir, f"ckpt_{step}.pt"))

    env.close()
    eval_env.close()
    for lg in (ep_log, upd_log, val_log):
        lg.close()
    open(os.path.join(cfg.out_dir, "done"), "w").close()
    return cfg.out_dir
