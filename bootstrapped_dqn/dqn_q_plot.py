import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN
import gymnasium as gym

# Load trained model
model = DQN.load("models/dqn_cartpole")

# Create environment
env = gym.make("CartPole-v1")

q_values_list = []
step_list = []

# Reset environment
obs, _ = env.reset()
done = False
step = 0

# Run one episode and collect Q values
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        Q_values = model.q_net(obs_tensor).detach().numpy().squeeze(0)
    
    action = np.argmax(Q_values)

    q_values_list.append(Q_values)
    step_list.append(step)

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    step += 1

# Convert to NumPy array
q_values_array = np.array(q_values_list)

# Plot Q values over time
plt.figure(figsize=(8, 5))
plt.plot(step_list, q_values_array[:, 0], label="Q(Left)")
plt.plot(step_list, q_values_array[:, 1], label="Q(Right)")
plt.xlabel("Time Step")
plt.ylabel("Q Value")
plt.title("Q Value Evolution in CartPole-v1")
plt.legend()
plt.grid()

# Save the plot
plt.savefig("q_values_plot.png")
print("Q-value plot saved as q_values_plot.png")

# Close environment
env.close()
