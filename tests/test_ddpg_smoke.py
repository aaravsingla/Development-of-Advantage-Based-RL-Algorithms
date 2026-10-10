"""DDPG/TD3 run end-to-end and log the value-estimation measurement."""

import pandas as pd
import pytest

from rl.ddpg import DDPGConfig, train


@pytest.mark.parametrize("algo", ["ddpg", "td3"])
def test_ddpg_trains_and_logs_value_estimates(tmp_path, algo):
    cfg = DDPGConfig(env_id="Pendulum-v1", algo=algo, total_steps=3000, learning_starts=1000,
                     eval_every=1500, eval_episodes=2, checkpoint_every=3000, out_dir=str(tmp_path / algo))
    assert cfg.twin_q == (algo == "td3") and cfg.policy_delay == (2 if algo == "td3" else 1)
    out = train(cfg)
    v = pd.read_csv(f"{out}/value_est.csv")
    # Pendulum only ends by a 200-step time limit (< the 459-step exclusion window), so no
    # state has a valid Monte Carlo target and the value metrics are correctly absent.
    assert len(v) == 2 and (v.n_states == 0).all() and v.eval_return.notna().all()
