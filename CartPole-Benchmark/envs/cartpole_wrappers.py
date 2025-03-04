import gymnasium as gym
import numpy as np

class PartialObservationWrapper(gym.ObservationWrapper):
    """
    Hides velocity (v, ω) so the agent can only observe position x and pole angle θ
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8, -0.418], dtype=np.float32),
            high=np.array([4.8, 0.418], dtype=np.float32),
            dtype=np.float32
        )

    def observation(self, obs):
        return np.array([obs[0],obs[2]], dtype=np.float32)
    
class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise to position x and pole angle θ to simulate sensor errors.
    """

    def __init__(self, env, noise_std = 0.05):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noisy_obs = obs.copy()
        noisy_obs[0] += np.random.normal(0,self.noise_std)
        noisy_obs[2] += np.random.normal(0,self.noise_std)
        return noisy_obs

class PartialObsWithNoiseWrapper(gym.ObservationWrapper):
    """
    Combines PartialObservationWrapper and ObservationNoiseWrapper
    """

    def __init__(self, env, noise_std=0.05):
        super().__init__(env)
        self.noise_std = noise_std
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8, -0.418], dtype=np.float32),
            high=np.array([4.8, 0.418], dtype=np.float32),
            dtype=np.float32
        )
    
    def observation(self, obs):
        noisy_obs = np.array([obs[0],obs[2]],dtype=np.float32)
        noisy_obs += np.random.normal(0, self.noise_std, size=noisy_obs.shape)
        return noisy_obs