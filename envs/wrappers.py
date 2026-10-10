"""Environment construction and a minimal, transparent vectorised env."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import envs.safe_nav  # noqa: F401  (registers SafeNav-v0)


class RewardNoise(gym.Wrapper):
    """Adds i.i.d. Gaussian noise N(0, sigma^2) to every reward.

    The clean reward is kept in info["clean_reward"] so learning curves are always
    reported on the true task reward, never the corrupted training signal.
    """

    def __init__(self, env: gym.Env, sigma: float, seed: int | None = None):
        super().__init__(env)
        self.sigma = sigma
        self.noise_rng = np.random.default_rng(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["clean_reward"] = reward
        if self.sigma > 0:
            reward = reward + self.sigma * self.noise_rng.standard_normal()
        return obs, reward, terminated, truncated, info


class SafetyCost(gym.Wrapper):
    """Adds a safety cost c_t and trains on r_t - w * c_t (fixed-weight penalty shaping).

    cost_type "velocity": c_t = 1 if forward velocity > threshold, the Safety-Gymnasium
    velocity constraint (SafetyHopperVelocity uses threshold 0.7402 m/s).
    info["task_reward"] keeps the unpenalised reward and info["cost"] the cost.
    """

    def __init__(self, env: gym.Env, cost_type: str, weight: float, threshold: float):
        super().__init__(env)
        self.cost_type, self.weight, self.threshold = cost_type, weight, threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.cost_type == "velocity":
            cost = float(info["x_velocity"] > self.threshold)
        else:
            raise ValueError(self.cost_type)
        info["task_reward"], info["cost"] = reward, cost
        return obs, reward - self.weight * cost, terminated, truncated, info


class RewardMultiplier(gym.Wrapper):
    """r -> c * r. Rescales returns and advantages without changing the optimal policy."""

    def __init__(self, env: gym.Env, c: float):
        super().__init__(env)
        self.c = c

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.setdefault("task_reward", reward)
        return obs, self.c * reward, terminated, truncated, info


def make_env(env_id: str, reward_noise: float = 0.0, seed: int | None = None, reward_multiplier: float = 1.0,
             cost_type: str = "", cost_weight: float = 0.0, cost_threshold: float = 0.7402,
             env_kwargs: dict | None = None) -> gym.Env:
    env = gym.make(env_id, **(env_kwargs or {}))
    env = gym.wrappers.ClipAction(env)
    if cost_type:
        env = SafetyCost(env, cost_type, cost_weight, cost_threshold)
    if reward_multiplier != 1.0:
        env = RewardMultiplier(env, reward_multiplier)
    env = RewardNoise(env, reward_noise, seed=seed)
    return env


class VecEnv:
    """Steps N independent envs and auto-resets them.

    Unlike gymnasium's built-in vector envs (whose autoreset semantics changed between
    versions), this returns the *true* next observation of every transition explicitly,
    which is exactly what GAE needs to bootstrap correctly on time-limit truncation.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, reward_noise: float = 0.0, **env_kwargs):
        self.envs = [make_env(env_id, reward_noise, seed=seed * 1000 + i, **env_kwargs) for i in range(num_envs)]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self._seed = seed
        self.ep_return = np.zeros(num_envs)  # task return: no noise, penalty or multiplier
        self.ep_cost = np.zeros(num_envs)
        self.ep_success = np.zeros(num_envs)
        self.last_costs = np.zeros(num_envs)  # per-step cost of the latest step (for Lagrangian PPO)
        self.ep_length = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> np.ndarray:
        obs = [env.reset(seed=self._seed * 1000 + i)[0] for i, env in enumerate(self.envs)]
        for i, env in enumerate(self.envs):
            env.action_space.seed(self._seed * 1000 + i)
        self.ep_return[:] = 0
        self.ep_cost[:] = 0
        self.ep_success[:] = 0
        self.ep_length[:] = 0
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        """Returns obs (post-reset), rewards, terminated, truncated, true_next_obs, finished.

        finished: list of (env_index, task_return, length, episode_cost, success) for episodes that ended.
        """
        obs_out, rews, terms, truncs, next_obs, finished = [], [], [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = env.step(a)
            self.ep_return[i] += info.get("task_reward", info.get("clean_reward", r))
            self.ep_cost[i] += info.get("cost", 0.0)
            self.last_costs[i] = info.get("cost", 0.0)
            self.ep_success[i] = max(self.ep_success[i], info.get("success", 0.0))
            self.ep_length[i] += 1
            next_obs.append(o)
            if term or trunc:
                finished.append((i, float(self.ep_return[i]), int(self.ep_length[i]), float(self.ep_cost[i]),
                                 float(self.ep_success[i])))
                self.ep_return[i] = 0
                self.ep_cost[i] = 0
                self.ep_success[i] = 0
                self.ep_length[i] = 0
                o, _ = env.reset()
            obs_out.append(o)
            rews.append(r)
            terms.append(term)
            truncs.append(trunc)
        return (
            np.stack(obs_out),
            np.array(rews, dtype=np.float64),
            np.array(terms, dtype=np.float64),
            np.array(truncs, dtype=np.float64),
            np.stack(next_obs),
            finished,
        )

    def close(self):
        for env in self.envs:
            env.close()
