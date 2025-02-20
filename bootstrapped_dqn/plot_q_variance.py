import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
import gymnasium as gym
import torch

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Create environment
env = gym.make("CartPole-v1")

# Load trained model
model = DQN.load("models/dqn_cartpole")

# Reset environment and extract observation
obs = env.reset()[0]  # Ensure correct extraction

if obs is None:
    raise ValueError("Error: env.reset() returned None instead of an observation.")

q_values_list = []

# Convert obs to tensor for model input
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

# Run the agent and record Q-values
for _ in range(1000):
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor)  # Get Q-values from Q-network
    q_values_list.append(q_values.numpy()[0])  # Convert to NumPy

    action = q_values.argmax().item()  # Choose action with highest Q-value
    obs, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs = env.reset()[0]  # Reset environment if episode ends

env.close()

# Compute variance of Q-values
q_values_array = np.array(q_values_list)
q_variance = np.var(q_values_array, axis=1)

# Plot the Q-value variance over time
plt.plot(q_variance)
plt.xlabel("Time Step")
plt.ylabel("Q-value Variance")
plt.title("Bootstrapped DQN: Q-value Uncertainty Over Time")

plt.savefig("results/q_value_variance.png")
plt.show()
