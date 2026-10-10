"""End-to-end smoke tests: training runs, writes its logs, and the probes read its checkpoints."""

import numpy as np
import pandas as pd
import torch

from instrumentation.probe import EpisodeCollector, batched_policy_gradients, load_agent
from rl.ppo import PPOConfig, train


def test_ppo_trains_and_probe_gradient_matches_autograd(tmp_path):
    out = train(PPOConfig(env_id="Pendulum-v1", total_steps=4096, num_steps=1024, checkpoint_every=2048,
                          out_dir=str(tmp_path / "run")))
    upd = pd.read_csv(f"{out}/updates.csv")
    assert len(upd) == 4 and np.isfinite(upd.kl_full).all()
    assert len(pd.read_csv(f"{out}/episodes.csv")) > 0

    agent = load_agent(f"{out}/ckpt_4096.pt")
    eps = EpisodeCollector("Pendulum-v1", 2).collect(agent, seed=0)
    obs = np.concatenate([e.obs for e in eps])
    act = np.concatenate([e.act for e in eps])
    w = np.random.default_rng(0).normal(size=(3, len(obs)))

    batched = batched_policy_gradients(agent.actor, obs, act, w)
    for k in range(3):  # compare against a plain one-at-a-time backward pass
        agent.actor.zero_grad()
        lp = agent.actor.log_prob(torch.as_tensor(obs), torch.as_tensor(act))
        (lp * torch.as_tensor(w[k], dtype=lp.dtype)).sum().backward()
        ref = torch.cat([p.grad.reshape(-1) for p in agent.actor.parameters()]).numpy()
        np.testing.assert_allclose(batched[k], ref, rtol=1e-4, atol=1e-6)
