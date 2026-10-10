"""Generalized Advantage Estimation (Schulman et al., 2016), written from scratch.

Notation (all arrays shaped [T, N]: T time steps, N parallel envs):
    rewards[t]      r_t
    values[t]       V(s_t)
    next_values[t]  V(s_{t+1}), where s_{t+1} is the *true* next observation.
                    On a time-limit truncation this is the final observation of the
                    episode (before the reset), not the first obs of the next episode.
                    At the last step of the rollout it is the bootstrap V(s_T).
    terminated[t]   1 if s_{t+1} is a true terminal state (value 0 by definition).
    episode_end[t]  1 if the episode ended at t for *any* reason (terminated OR truncated).

TD error:     delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
Advantage:    A_t = delta_t + gamma * lam * (1 - episode_end_t) * A_{t+1}

Two flags are needed because the two ways an episode can end mean different things:
  * termination: the future is worth 0, so do not bootstrap and cut the recursion.
  * truncation (time limit / rollout cut): the future is NOT worth 0, so bootstrap from
    V(s_{t+1}), but still cut the recursion since A_{t+1} belongs to another episode.

Special cases:
    lam = 0  ->  A_t = delta_t                                  (1-step TD, low variance, biased by V)
    lam = 1  ->  A_t = sum_l gamma^l r_{t+l} + bootstrap - V(s_t)  (Monte Carlo, unbiased, high variance)
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_end: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (advantages, returns), both shaped like `rewards`.

    returns = advantages + values are the TD(lambda) targets used to fit the critic.
    Accepts 1-D arrays (single env) as well as [T, N].
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    squeeze = rewards.ndim == 1
    if squeeze:
        rewards, values, next_values, terminated, episode_end = (
            np.asarray(x, dtype=np.float64)[:, None]
            for x in (rewards, values, next_values, terminated, episode_end)
        )
    else:
        values, next_values, terminated, episode_end = (
            np.asarray(x, dtype=np.float64) for x in (values, next_values, terminated, episode_end)
        )

    T = rewards.shape[0]
    not_term = 1.0 - terminated
    not_end = 1.0 - episode_end

    deltas = rewards + gamma * not_term * next_values - values
    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(T)):
        last = deltas[t] + gamma * lam * not_end[t] * last
        advantages[t] = last

    returns = advantages + values
    if squeeze:
        return advantages[:, 0], returns[:, 0]
    return advantages, returns


def td_errors(rewards, values, next_values, terminated, gamma):
    """delta_t, exposed separately for tests and analysis."""
    return np.asarray(rewards) + gamma * (1.0 - np.asarray(terminated)) * np.asarray(next_values) - np.asarray(values)


def reward_noise_variance_multiplier(gamma: float, lam: float) -> float:
    """Var of the reward-noise component of A_t, per unit sigma^2, infinite horizon.

    If r_t carries i.i.d. noise eps_t ~ N(0, sigma^2) and V is held fixed, then
    A_t contains sum_l (gamma*lam)^l eps_{t+l}, whose variance is
    sigma^2 * sum_l (gamma*lam)^{2l} = sigma^2 / (1 - (gamma*lam)^2).
    """
    return 1.0 / (1.0 - (gamma * lam) ** 2)
