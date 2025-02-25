import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import realworldrl_suite.environments as rwrl
from collections import deque
from torch import nn, optim

# ========================== 1️⃣ Set Up Environment ==========================

# Choose the environment (walker, cartpole, humanoid, etc.)
ENV_NAME = "walker"

# Create environment with real-world factors enabled
env = rwrl.create(
    ENV_NAME,
    realworld_kwargs={
        "enable_all": True  # Enables all real-world challenges (sensor noise, delays, etc.)
    },
)

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]  # Continuous action space

# ========================== 2️⃣ Define DQN Model ==========================

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ========================== 3️⃣ Training Hyperparameters ==========================

GAMMA = 0.99
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TARGET_UPDATE = 10
EPISODES = 500
MAX_STEPS = 1000

# Initialize DQN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Experience replay
memory = deque(maxlen=MEMORY_SIZE)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            return policy_net(state).cpu().numpy()  # Continuous action

# ========================== 4️⃣ Training Loop ==========================

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
epsilon = 1.0
reward_list = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for step in range(MAX_STEPS):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

        # Train DQN using experience replay
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)  # Continuous actions
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states)
            next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = loss_fn(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save episode reward
    reward_list.append(total_reward)
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    print(f"Episode {episode}, Reward: {total_reward}")

# Save trained model
torch.save(policy_net.state_dict(), "../models/walker_dqn.pth")

# Save training reward plot
plt.plot(reward_list)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Walker - Training Reward")
plt.savefig("../results/walker_training_reward.png")
plt.close()
