"""Running statistics for observation normalisation and reward scaling."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running mean / variance."""

    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = np.array(d["mean"]), np.array(d["var"]), float(d["count"])


class ObsNormalizer:
    def __init__(self, shape, clip: float = 10.0):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.frozen = False

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update and not self.frozen:
            self.rms.update(obs)
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip).astype(np.float32)


class RewardScaler:
    """Divides rewards by the running std of the discounted return (as in PPO's MuJoCo setup).

    This is a pure rescaling (no mean shift), so it does not change the optimal policy.
    `scale` is exposed so value predictions can be mapped back to raw-reward units.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0):
        self.rms = RunningMeanStd(())
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.clip = clip
        self.frozen = False

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.rms.var + 1e-8))

    def __call__(self, rewards: np.ndarray, episode_end: np.ndarray) -> np.ndarray:
        if not self.frozen:
            self.ret = self.ret * self.gamma + rewards
            self.rms.update(self.ret)
            self.ret[episode_end.astype(bool)] = 0.0
        return np.clip(rewards / self.scale, -self.clip, self.clip)
