import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment
env = gym.make("CartPole-v1")

# Initialize DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=500000)

# Save the trained model
model.save("models/dqn_cartpole")

print("Model training completed and saved!")