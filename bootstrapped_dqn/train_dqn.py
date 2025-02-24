import os
import torch
import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_timesteps", type=int, default=500000, help="Total training steps")
parser.add_argument("--device", type=str, default="auto", help="Choose device: auto, cpu, cuda")
parser.add_argument("--logdir", type=str, default="logs", help="Directory to save logs")
parser.add_argument("--save_path", type=str, default="models/dqn_cartpole", help="Model save path")
parser.add_argument("--num_eval_episodes", type=int, default=10, help="Number of evaluation episodes")
args = parser.parse_args()

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

# Ensure directories exist
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# Configure TensorBoard logging
logger = configure(args.logdir, ["tensorboard", "stdout"])

# Create environment
env = gym.make("CartPole-v1")

# Initialize DQN model (Hugging Face settings)
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.002,
    buffer_size=200000,  # Prevents forgetting
    batch_size=64,
    learning_starts=1000,
    target_update_interval=250,  # More stable updates
    train_freq=256,
    gradient_steps=32,  # Reduces aggressive updates
    gamma=0.99,
    exploration_fraction=0.3,  # Extend exploration phase
    exploration_final_eps=0.1,  # Keep more exploration
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="logs/tensorboard/",
    device=device
)


model.set_logger(logger)

# Train with periodic checkpoints
checkpoint_interval = args.total_timesteps // 10
for i in range(1, 11):
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
    model.save(f"{args.save_path}_checkpoint_{i}")

# Save final model
model.save(args.save_path)
print("Training complete. Model saved.")

# Evaluate the trained model
print("\nEvaluating the trained model...")
model = DQN.load(args.save_path, env=env)

total_rewards = []
for episode in range(args.num_eval_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Save evaluation results
avg_reward = sum(total_rewards) / len(total_rewards)
with open("logs/evaluation_results.txt", "a") as f:
    f.write(f"Model: {args.save_path}, Episodes: {args.num_eval_episodes}, Avg Reward: {avg_reward:.2f}\n")

print(f"Evaluation complete. Results saved to logs/evaluation_results.txt")
env.close()
