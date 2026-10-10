"""GAE is checked against a direct, non-recursive implementation of its definition."""

import numpy as np
import pytest

from rl.gae import compute_gae, reward_noise_variance_multiplier, td_errors


def brute_force_gae(rewards, values, next_values, terminated, episode_end, gamma, lam):
    """A_t = sum_{l>=0} (gamma*lam)^l delta_{t+l}, summing only until the episode ends."""
    T = len(rewards)
    deltas = td_errors(rewards, values, next_values, terminated, gamma)
    adv = np.zeros(T)
    for t in range(T):
        total, coef = 0.0, 1.0
        for k in range(t, T):
            total += coef * deltas[k]
            if episode_end[k]:
                break
            coef *= gamma * lam
        adv[t] = total
    return adv


def random_rollout(rng, T=50, p_term=0.05, p_trunc=0.03):
    rewards = rng.normal(size=T)
    values = rng.normal(size=T)
    next_values = np.append(values[1:], rng.normal())
    terminated = (rng.random(T) < p_term).astype(float)
    truncated = ((rng.random(T) < p_trunc) & (terminated == 0)).astype(float)
    # After a truncation the next stored value belongs to a new episode, so the true
    # next value is some other number (V of the final obs before reset).
    next_values = np.where(truncated == 1, rng.normal(size=T), next_values)
    episode_end = np.maximum(terminated, truncated)
    return rewards, values, next_values, terminated, episode_end


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9, 0.95, 1.0])
@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force(lam, seed):
    rng = np.random.default_rng(seed)
    args = random_rollout(rng)
    adv, ret = compute_gae(*args, gamma=0.99, lam=lam)
    np.testing.assert_allclose(adv, brute_force_gae(*args, 0.99, lam), atol=1e-10)
    np.testing.assert_allclose(ret, adv + args[1], atol=1e-10)


def test_lambda_zero_is_td_error():
    rng = np.random.default_rng(0)
    args = random_rollout(rng)
    adv, _ = compute_gae(*args, gamma=0.99, lam=0.0)
    np.testing.assert_allclose(adv, td_errors(*args[:4], 0.99))


def test_lambda_one_is_monte_carlo_minus_baseline():
    """Single episode that terminates: A_t = sum_k gamma^k r_{t+k} - V(s_t), no V(s_{t+1}) anywhere."""
    gamma = 0.97
    rewards = np.array([1.0, -2.0, 0.5, 3.0])
    values = np.array([0.3, -1.0, 2.0, 0.7])
    next_values = np.array([-1.0, 2.0, 0.7, 123.0])  # last entry must be ignored (terminal)
    terminated = np.array([0, 0, 0, 1.0])
    adv, ret = compute_gae(rewards, values, next_values, terminated, terminated, gamma, 1.0)
    mc = np.array([sum(gamma**k * rewards[t + k] for k in range(4 - t)) for t in range(4)])
    np.testing.assert_allclose(adv, mc - values)
    np.testing.assert_allclose(ret, mc)


def test_truncation_bootstraps_but_does_not_leak_across_episodes():
    gamma, lam = 0.9, 0.8
    # episode A: steps 0-1, truncated at 1 with V(final obs) = 5
    # episode B: steps 2-3, rollout ends at 3 (bootstrap V(s_4) = 2)
    rewards = np.array([1.0, 1.0, 0.0, 0.0])
    values = np.array([0.0, 0.0, 100.0, 0.0])
    next_values = np.array([0.0, 5.0, 0.0, 2.0])
    terminated = np.zeros(4)
    episode_end = np.array([0, 1.0, 0, 0])
    adv, _ = compute_gae(rewards, values, next_values, terminated, episode_end, gamma, lam)
    d1 = 1 + gamma * 5
    assert adv[1] == pytest.approx(d1)
    assert adv[0] == pytest.approx(1 + gamma * lam * d1)  # unaffected by V=100 in episode B
    assert adv[3] == pytest.approx(gamma * 2)


def test_termination_does_not_bootstrap():
    adv, _ = compute_gae([1.0], [0.0], [999.0], [1.0], [1.0], 0.99, 0.95)
    assert adv[0] == pytest.approx(1.0)


def test_vectorised_equals_per_env():
    rng = np.random.default_rng(1)
    cols = [random_rollout(rng) for _ in range(4)]
    stacked = [np.stack([c[k] for c in cols], axis=1) for k in range(5)]
    adv, _ = compute_gae(*stacked, gamma=0.99, lam=0.95)
    for j, c in enumerate(cols):
        np.testing.assert_allclose(adv[:, j], compute_gae(*c, gamma=0.99, lam=0.95)[0])


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_reward_noise_variance_formula(lam):
    """Empirical Var of the noise component of A_t matches sigma^2 / (1 - (gamma*lam)^2)."""
    gamma, sigma, T, n = 0.99, 0.7, 3000, 400
    rng = np.random.default_rng(0)
    eps = sigma * rng.standard_normal((T, n))
    adv, _ = compute_gae(eps, np.zeros((T, n)), np.zeros((T, n)), np.zeros((T, n)), np.zeros((T, n)), gamma, lam)
    emp = adv[: T // 3].var()  # far from the rollout end, so effectively infinite horizon
    theory = sigma**2 * reward_noise_variance_multiplier(gamma, lam)
    assert emp == pytest.approx(theory, rel=0.1)
