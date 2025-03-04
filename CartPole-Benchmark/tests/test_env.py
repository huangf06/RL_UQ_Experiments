import pytest
import numpy as np
from envs.make_env import make_cartpole

@pytest.mark.parametrize("version, expected_dim", [
    (0, 4),  # Original environment (x, v, θ, ω)
    (1, 2),  # Partial observability (x, θ)
    (2, 4),  # Observation noise (x, v, θ, ω)
    (3, 2)   # Partial observability + noise (x, θ)
])

def test_observation_space(version, expected_dim):
    """Ensure observation space matches expected dimension."""
    env = make_cartpole(version, noise_std=0.05)
    state, _ = env.reset()
    assert len(state) == expected_dim, f"Error: CartPole-{version} should have {expected_dim} dims, got {len(state)}"
    env.close()

@pytest.mark.parametrize("version", [2, 3])
def test_observation_noise(version):
    """Verify that noise is applied to the observations."""
    env = make_cartpole(version, noise_std=0.1)

    states = np.array([env.reset()[0] for _ in range(5)])
    std_deviation = np.std(states, axis=0)
    
    assert np.any(std_deviation > 0), f"Error: CartPole-{version} does not apply noise correctly"
    env.close()

def test_env_step():
    """Check that step() returns values following Gymnasium conventions."""
    env = make_cartpole(0)
    state, _ = env.reset()

    action = env.action_space.sample()
    next_state, reward, done, truncated, _ = env.step(action)

    assert isinstance(next_state, np.ndarray), "Error: next_state should be a NumPy array"
    assert isinstance(reward, float), "Error: reward should be a float"
    assert isinstance(done, bool), "Error: done should be a boolean"
    assert isinstance(truncated, bool), "Error: truncated should be a boolean"

    env.close()
