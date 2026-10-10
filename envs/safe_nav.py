"""SafeNav-v0: 2D point-robot navigation among circular obstacles, with CBF tools.

Dynamics (single integrator):  p_{t+1} = clip(p_t + dt * u_t, arena),   u in [-1, 1]^2
Episode ends on: collision (terminated), reaching the goal (terminated), 200 steps (truncated).
Observation (20-d): goal offset / L (2), position / L (2), 16 lidar ranges / max_range.

Control barrier function for obstacle i with safety radius R_i = r_i + margin:
    h_i(p) = (||p - c_i||^2 - R_i^2) / R_i^2          (h_i >= 0  <=>  outside the safety radius)
Discrete-time CBF condition (forward invariance of {h >= 0}):
    h_i(p_{t+1}) >= (1 - alpha) * h_i(p_t),   0 < alpha <= 1
Far from an obstacle h is large, so the condition is slack; it only binds close to the
obstacle, where it limits how fast the robot may approach. A collision (h_i < -margin-term,
i.e. inside r_i) is therefore preceded by several steps of CBF violation, which is what lets
CBF shaping give a dense warning signal instead of a single sparse crash penalty.

Reward modes (all share the task reward  r_task = 10 * (d_{t} - d_{t+1}) + 10 * success):
    task        r_task                                                (collision only ends the episode)
    sparse      r_task - coef * 1[collision]
    distance    r_task - coef * sum_i max(0, 1 - dist_to_surface_i / d_safe)
    cbf         r_task - coef * sum_i max(0, (1 - alpha) h_i(p_t) - h_i(p_{t+1}))
info: task_reward, cost (= 1 on collision), success, h_min, cbf_violation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from scipy.optimize import minimize


class SafeNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_mode: str = "task", coef: float = 0.0, alpha: float = 0.2, margin: float = 0.1,
                 d_safe: float = 0.5, n_obstacles: int = 8, arena: float = 3.0, dt: float = 0.1,
                 max_steps: int = 200, n_rays: int = 16, max_range: float = 2.0, goal_tol: float = 0.2):
        assert reward_mode in ("task", "sparse", "distance", "cbf")
        self.reward_mode, self.coef, self.alpha, self.margin, self.d_safe = reward_mode, coef, alpha, margin, d_safe
        self.n_obs, self.L, self.dt, self.max_steps = n_obstacles, arena, dt, max_steps
        self.n_rays, self.max_range, self.goal_tol = n_rays, max_range, goal_tol
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        self.ray_dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4 + n_rays,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    # ------------------------------------------------------------------ layout
    def _sample_layout(self):
        rng = self.np_random
        L = self.L
        for _ in range(1000):
            centers, radii = [], []
            while len(centers) < self.n_obs:
                c = rng.uniform(-L + 0.5, L - 0.5, 2)
                r = rng.uniform(0.3, 0.6)
                if all(np.linalg.norm(c - c2) > r + r2 + 0.4 for c2, r2 in zip(centers, radii)):
                    centers.append(c)
                    radii.append(r)
            self.centers, self.radii = np.array(centers), np.array(radii)
            start = rng.uniform(-L + 0.3, L - 0.3, 2)
            goal = rng.uniform(-L + 0.3, L - 0.3, 2)
            clear = lambda p: np.all(np.linalg.norm(self.centers - p, axis=1) - self.radii > 0.4)
            if np.linalg.norm(goal - start) > 3.0 and clear(start) and clear(goal):
                self.pos, self.goal = start, goal
                return
        raise RuntimeError("could not sample a layout")

    # ------------------------------------------------------------------ CBF
    def h(self, p: np.ndarray) -> np.ndarray:
        R2 = (self.radii + self.margin) ** 2
        return (np.sum((p - self.centers) ** 2, axis=1) - R2) / R2

    def h_grad(self, p: np.ndarray) -> np.ndarray:
        R2 = (self.radii + self.margin) ** 2
        return 2 * (p - self.centers) / R2[:, None]

    def cbf_filter(self, u_ref: np.ndarray, alpha_c: float | None = None) -> np.ndarray:
        """Minimally modify u_ref so that every obstacle satisfies the CBF condition:
            min ||u - u_ref||^2   s.t.  grad h_i(p) . (dt u) >= -alpha_c h_i(p)  for all i,   |u| <= 1
        u = 0 always satisfies the constraints whenever h >= 0, so the QP is always feasible
        for this single-integrator robot (a property worth stating in an interview).
        The linearisation drops the O(dt^2 |u|^2) term of h, which is >= 0 (h is convex in p),
        so the true discrete-time h decreases no faster than the constraint allows.
        """
        alpha_c = self.alpha if alpha_c is None else alpha_c
        u_ref = np.clip(np.asarray(u_ref, dtype=np.float64), -1, 1)
        hv, G = self.h(self.pos), self.h_grad(self.pos) * self.dt
        near = hv < 5.0  # far obstacles cannot become active within one step
        if not near.any() or np.all(G[near] @ u_ref >= -alpha_c * hv[near]):
            return u_ref
        A, b = G[near], -alpha_c * hv[near]
        res = minimize(lambda u: np.sum((u - u_ref) ** 2), x0=np.zeros(2), jac=lambda u: 2 * (u - u_ref),
                       bounds=[(-1, 1), (-1, 1)], method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda u: A @ u - b, "jac": lambda u: A}])
        return res.x if res.success else np.zeros(2)

    # ------------------------------------------------------------------ obs
    def _lidar(self) -> np.ndarray:
        oc = self.centers - self.pos  # [K, 2]
        proj = self.ray_dirs @ oc.T  # [R, K]  distance along ray to closest approach
        perp2 = np.sum(oc**2, 1)[None] - proj**2
        disc = self.radii[None] ** 2 - perp2
        hit = (disc >= 0) & (proj > 0)
        dist = np.where(hit, proj - np.sqrt(np.maximum(disc, 0)), np.inf)
        d = np.clip(dist.min(1), 0, self.max_range)
        return d / self.max_range

    def _obs(self) -> np.ndarray:
        return np.concatenate([(self.goal - self.pos) / self.L, self.pos / self.L, self._lidar()]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_layout()
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        h_prev = self.h(self.pos)
        d_prev = np.linalg.norm(self.goal - self.pos)
        self.pos = np.clip(self.pos + self.dt * u, -self.L, self.L)
        self.t += 1
        h_new = self.h(self.pos)
        d_new = np.linalg.norm(self.goal - self.pos)
        surface = np.linalg.norm(self.pos - self.centers, axis=1) - self.radii

        collision = bool(np.any(surface < 0))
        success = (not collision) and d_new < self.goal_tol
        task = 10.0 * (d_prev - d_new) + 10.0 * success
        violation = np.maximum(0.0, (1 - self.alpha) * h_prev - h_new)

        if self.reward_mode == "task":
            reward = task
        elif self.reward_mode == "sparse":
            reward = task - self.coef * collision
        elif self.reward_mode == "distance":
            reward = task - self.coef * np.sum(np.maximum(0.0, 1 - surface / self.d_safe))
        else:
            reward = task - self.coef * violation.sum()

        terminated = collision or success
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"task_reward": task, "cost": float(collision), "success": float(success),
                "h_min": float(h_new.min()), "cbf_violation": float(violation.sum())}
        return self._obs(), float(reward), terminated, truncated, info


gym.register(id="SafeNav-v0", entry_point="envs.safe_nav:SafeNavEnv")
