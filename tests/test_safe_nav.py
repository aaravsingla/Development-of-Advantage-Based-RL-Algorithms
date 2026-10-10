"""SafeNav geometry, CBF definitions, and the safety filter's guarantee."""

import gymnasium as gym
import numpy as np
import pytest

import envs.safe_nav  # noqa: F401
from envs.safe_nav import SafeNavEnv


def make(**kw) -> SafeNavEnv:
    env = SafeNavEnv(**kw)
    env.reset(seed=0)
    return env


def test_h_sign_matches_safety_radius():
    env = make()
    c, R = env.centers[0], env.radii[0] + env.margin
    assert env.h(c + np.array([R * 1.01, 0]))[0] > 0
    assert env.h(c + np.array([R * 0.99, 0]))[0] < 0
    # analytic gradient vs finite differences
    p, eps = env.pos.copy(), 1e-6
    fd = np.stack([(env.h(p + e) - env.h(p - e)) / (2 * eps) for e in np.eye(2) * eps], 1)
    np.testing.assert_allclose(env.h_grad(p), fd, rtol=1e-5, atol=1e-6)


def test_lidar_matches_brute_force_ray_march():
    env = make()
    lid = env._lidar() * env.max_range
    for k, d in enumerate(env.ray_dirs):
        ts = np.linspace(0, env.max_range, 20001)
        pts = env.pos + ts[:, None] * d
        inside = (np.linalg.norm(pts[:, None] - env.centers[None], axis=2) < env.radii[None]).any(1)
        expected = ts[inside.argmax()] if inside.any() else env.max_range
        assert lid[k] == pytest.approx(expected, abs=1e-3)


def test_collision_terminates_with_cost():
    env = make(reward_mode="sparse", coef=5.0)
    env.pos = env.centers[0] + np.array([env.radii[0] + 0.05, 0])
    _, r, term, trunc, info = env.step(np.array([-1.0, 0.0]))
    assert term and not trunc and info["cost"] == 1.0 and info["success"] == 0.0
    assert r == pytest.approx(info["task_reward"] - 5.0)


def test_cbf_penalty_is_dense_before_collision():
    """Driving straight at an obstacle is penalised for several steps before impact."""
    env = make(reward_mode="cbf", coef=1.0)
    c, r = env.centers[0], env.radii[0]
    env.pos = c + np.array([r + 1.0, 0.0])
    penalised_steps = 0
    for _ in range(30):
        _, _, term, _, info = env.step(np.array([-1.0, 0.0]))
        penalised_steps += info["cbf_violation"] > 0
        if term:
            break
    assert info["cost"] == 1.0 and penalised_steps >= 3


def test_cbf_filter_prevents_all_collisions_for_random_policy():
    env = gym.make("SafeNav-v0").unwrapped
    rng = np.random.default_rng(0)
    collisions, steps = 0, 0
    for ep in range(30):
        env.reset(seed=ep)
        for _ in range(200):
            u = env.cbf_filter(rng.uniform(-1, 1, 2))
            _, _, term, trunc, info = env.step(u)
            collisions += info["cost"]
            steps += 1
            assert info["h_min"] > -1e-6  # the safe set {h >= 0} is never left
            if term or trunc:
                break
    assert collisions == 0 and steps > 1000


def test_filter_is_identity_when_safe():
    env = make()
    env.pos = np.array([10.0, 10.0])  # far from everything (outside arena is fine for this check)
    u = np.array([0.3, -0.7])
    np.testing.assert_allclose(env.cbf_filter(u), u)
