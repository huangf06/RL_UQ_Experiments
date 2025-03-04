import gymnasium as gym

def make_cartpole_base(render_mode=None, seed=None, max_episode_steps=None):
    """
    Create a standard CartPole-v1 environment with optional configurations.

    Parameters:
        render_mode (str, optional): Rendering mode ('human', 'rgb_array', or None).
        seed (int, optional): Random seed for reproducibility.
        max_episode_steps (int, optional): Maximum steps per episode.

    Returns:
        gym.Env: Configured CartPole-v1 environment.
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    
    if seed is not None:
        env.reset(seed=seed)
    
    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps

    return env
