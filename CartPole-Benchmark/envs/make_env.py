from envs.cartpole_base import make_cartpole_base
from envs.cartpole_wrappers import PartialObservationWrapper, ObservationNoiseWrapper, PartialObsWithNoiseWrapper

def make_cartpole(version, noise_std=0.05, render_mode=None, seed=None, max_episode_steps=None):
    """
    Create a CartPole environment with different observation modifications.

    Parameters:
        version (int): 
            - 0: Standard CartPole
            - 1: Partial Observability (Hides velocity v, ω)
            - 2: Observation Noise (Adds Gaussian noise to x, θ)
            - 3: Partial Observability + Noise
        noise_std (float, optional): Standard deviation of noise
        render_mode (str, optional): Rendering mode ('human', 'rgb_array', or None)
        seed (int, optional): Random seed for reproducibility.
        max_episode_steps (init, optional): Max steps per episode.

    Returns: 
        gym.Env: Configured CartPole environment.
    """

    env = make_cartpole_base(render_mode=render_mode, seed=seed, max_episode_steps=max_episode_steps)

    if version == 1:
        env = PartialObservationWrapper(env)
    elif version == 2:
        env = ObservationNoiseWrapper(env, noise_std=noise_std)
    elif version == 3:
        env = PartialObsWithNoiseWrapper(env, noise_std=noise_std)

    return env