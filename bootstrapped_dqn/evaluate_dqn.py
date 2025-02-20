import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# Recreate the environment (for visualization)
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load the trained model
model = DQN.load("models/dqn_cartpole")

obs, _ = env.reset()

# Record cart positions and pole angles over time
cart_positions = []
pole_angles = []

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)  # Use trained policy
    obs, _, terminated, truncated, _ = env.step(action)
    
    cart_pos = obs[0]  # Cart position
    pole_angle = obs[2]  # Pole angle
    cart_positions.append(cart_pos)
    pole_angles.append(pole_angle)

    if terminated or truncated:
        break

env.close()

# Plot the cart position and pole angle over time··
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cart_positions)
plt.xlabel("Time Step")
plt.ylabel("Cart Position")
plt.title("Cart Position Over Time")

plt.subplot(1, 2, 2)
plt.plot(pole_angles)
plt.xlabel("Time Step")
plt.ylabel("Pole Angle (radians)")
plt.title("Pole Angle Over Time")

plt.savefig("results/cartpole_trajectory.png")
plt.show()
