"""Tools for measuring estimator properties on a *frozen* policy and critic.

Why frozen: to isolate the effect of the advantage estimator, every lambda must be
evaluated on the same policy, the same critic, and (paired) the same trajectories.
Then any difference in the resulting gradients is caused by the estimator alone.

Batches are made of complete episodes started from fresh resets. Stopping a batch "when
enough steps were collected" and dropping unfinished episodes would over-sample short
episodes, so a batch is always exactly `episodes_per_batch` full episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

import envs.safe_nav  # noqa: F401  (registers SafeNav-v0)
from rl.gae import compute_gae
from rl.networks import Critic, GaussianActor


@dataclass
class FrozenAgent:
    actor: GaussianActor
    critic: Critic
    obs_mean: np.ndarray
    obs_std: np.ndarray
    reward_scale: float
    reward_clip: float
    gamma: float
    env_id: str
    global_step: int
    config: dict

    def norm(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.obs_mean) / self.obs_std, -10, 10).astype(np.float32)

    def scale_rewards(self, r: np.ndarray) -> np.ndarray:
        return np.clip(r / self.reward_scale, -self.reward_clip, self.reward_clip)


def load_agent(path: str) -> FrozenAgent:
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    env = gym.make(cfg["env_id"])
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    actor, critic = GaussianActor(obs_dim, act_dim, cfg["hidden"]), Critic(obs_dim, cfg["hidden"])
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    actor.eval()
    critic.eval()
    if ck["obs_rms"] is not None:
        mean, std = np.array(ck["obs_rms"]["mean"]), np.sqrt(np.array(ck["obs_rms"]["var"]) + 1e-8)
    else:
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)
    scale = float(np.sqrt(ck["reward_rms"]["var"] + 1e-8)) if ck["reward_rms"] is not None else 1.0
    clip = 10.0 if ck["reward_rms"] is not None else np.inf
    return FrozenAgent(actor, critic, mean, std, scale, clip, cfg["gamma"], cfg["env_id"], ck["global_step"], cfg)


@dataclass
class Episode:
    obs: np.ndarray  # [L, obs_dim] normalised
    act: np.ndarray  # [L, act_dim]
    rew: np.ndarray  # [L] raw clean reward
    terminated: bool  # ended in a true terminal state (vs time-limit truncation)
    final_obs: np.ndarray  # [obs_dim] normalised obs after the last step


class EpisodeCollector:
    def __init__(self, env_id: str, n_envs: int):
        self.envs = [gym.wrappers.ClipAction(gym.make(env_id)) for _ in range(n_envs)]

    def collect(self, agent: FrozenAgent, seed: int, deterministic: bool = False) -> list[Episode]:
        """One full episode per env, run in lockstep (batched policy forward pass)."""
        n = len(self.envs)
        rng = torch.Generator().manual_seed(seed)
        obs = [self.envs[i].reset(seed=seed * 1000 + i)[0] for i in range(n)]
        bufs = [dict(obs=[], act=[], rew=[]) for _ in range(n)]
        active = list(range(n))
        episodes: list[Episode | None] = [None] * n
        while active:
            o = agent.norm(np.stack([obs[i] for i in active]))
            with torch.no_grad():
                dist = agent.actor.dist(torch.as_tensor(o))
                a = dist.mean if deterministic else dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=rng)
            a = a.numpy()
            still = []
            for j, i in enumerate(active):
                nxt, r, term, trunc, _ = self.envs[i].step(a[j])
                bufs[i]["obs"].append(o[j])
                bufs[i]["act"].append(a[j])
                bufs[i]["rew"].append(r)
                if term or trunc:
                    episodes[i] = Episode(
                        np.stack(bufs[i]["obs"]),
                        np.stack(bufs[i]["act"]).astype(np.float32),
                        np.array(bufs[i]["rew"]),
                        bool(term),
                        agent.norm(nxt[None])[0],
                    )
                else:
                    obs[i] = nxt
                    still.append(i)
            active = still
        return episodes

    def close(self):
        for e in self.envs:
            e.close()


def episode_values(agent: FrozenAgent, ep: Episode) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        v = agent.critic(torch.as_tensor(ep.obs)).numpy().astype(np.float64)
        v_final = 0.0 if ep.terminated else agent.critic(torch.as_tensor(ep.final_obs[None])).item()
    return v, v_final


def episode_advantages(agent: FrozenAgent, ep: Episode, v: np.ndarray, v_final: float,
                       lam: float, noise: np.ndarray | None = None) -> np.ndarray:
    """GAE on one complete episode. `noise` (raw reward units) is added before scaling."""
    raw = ep.rew if noise is None else ep.rew + noise
    r = agent.scale_rewards(raw)
    L = len(r)
    next_v = np.append(v[1:], v_final)
    term = np.zeros(L)
    term[-1] = float(ep.terminated)
    end = np.zeros(L)
    end[-1] = 1.0
    adv, _ = compute_gae(r, v, next_v, term, end, agent.gamma, lam)
    return adv


def actor_params(actor: GaussianActor) -> list[torch.nn.Parameter]:
    return [p for p in actor.parameters()]


def batched_policy_gradients(actor: GaussianActor, obs: np.ndarray, act: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """For each row w of `weights` ([K, M]) returns grad_theta sum_i w_i log pi(a_i|s_i), flattened -> [K, P].

    With w = A_hat / M this is the policy-gradient estimate (1/M) sum_i grad log pi(a_i|s_i) A_hat_i,
    which is exactly the gradient of PPO's clipped surrogate at theta = theta_old (ratio = 1,
    so the clip is inactive). One backward pass handles all K advantage estimators at once.
    """
    params = actor_params(actor)
    logp = actor.log_prob(torch.as_tensor(obs), torch.as_tensor(act))
    w = torch.as_tensor(weights, dtype=logp.dtype)
    grads = torch.autograd.grad(logp, params, grad_outputs=w, is_grads_batched=True)
    return torch.cat([g.reshape(w.shape[0], -1) for g in grads], dim=1).numpy().astype(np.float64)


def discounted_mc_returns(r: np.ndarray, gamma: float) -> np.ndarray:
    g = np.zeros_like(r, dtype=np.float64)
    acc = 0.0
    for t in reversed(range(len(r))):
        acc = r[t] + gamma * acc
        g[t] = acc
    return g
