import matplotlib.pyplot as plt
import numpy as np

# Load rewards
reward_list = np.loadtxt("../results/walker_rewards.txt")

# Plot
plt.plot(reward_list)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Walker - Training Reward")
plt.savefig("../results/walker_training_reward.png")
plt.show()