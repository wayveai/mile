import gym
import numpy as np
import torch

from mile.agents.rl_birdview.models.ppo_buffer import PpoBuffer


def test_sampled_advantages_are_normalized():
    observation_space = gym.spaces.Dict({'state': gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)})
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    buffer = PpoBuffer(3, observation_space, action_space)
    buffer.full = True
    buffer.advantages[:] = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    sample = next(buffer.get())
    assert abs(sample.advantages.mean().item()) < 1e-6
    torch.testing.assert_close(
        sample.advantages.std(unbiased=False), sample.advantages.new_tensor(1.0), rtol=1e-5, atol=1e-5,
    )


def test_constant_advantages_produce_finite_zero_samples():
    observation_space = gym.spaces.Dict({'state': gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)})
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    buffer = PpoBuffer(2, observation_space, action_space)
    buffer.full = True
    buffer.advantages[:] = 7.0
    sample = next(buffer.get())
    torch.testing.assert_close(sample.advantages, torch.zeros_like(sample.advantages))
