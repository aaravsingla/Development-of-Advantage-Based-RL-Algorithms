"""Train a single PPO run.

    python -m experiments.train_ppo --env Hopper-v5 --lam 0.95 --seed 0 --steps 1000000
"""

import argparse

import torch

from rl.ppo import PPOConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Hopper-v5")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--reward-noise", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.add_argument("--no-norm-adv", action="store_true")
    a = p.parse_args()

    torch.set_num_threads(1)
    out = a.out or f"results/runs/debug/{a.env}_lam{a.lam}_s{a.seed}"
    cfg = PPOConfig(
        env_id=a.env,
        gae_lambda=a.lam,
        seed=a.seed,
        total_steps=a.steps,
        reward_noise=a.reward_noise,
        norm_adv=not a.no_norm_adv,
        out_dir=out,
    )
    train(cfg)


if __name__ == "__main__":
    main()
