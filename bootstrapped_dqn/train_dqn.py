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
args = parser.parse_args()

# Automatically select device (GPU or CPU)
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

# Ensure the log and model directories exist
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# Configure logging for TensorBoard and stdout
logger = configure(args.logdir, ["tensorboard"])

# Create CartPole environment
env = gym.make("CartPole-v1")

# Initialize DQN model with tuned hyperparameters
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,  # Set a reasonable learning rate
    buffer_size=100000,  # Experience replay buffer size
    batch_size=64,  # Mini-batch size for training
    learning_starts=1000,  # Steps before training begins
    target_update_interval=500,  # Target network update frequency
    exploration_fraction=0.1,  # Fraction of total training steps used for exploration
    exploration_final_eps=0.01,  # Minimum exploration rate
    verbose=1,
    device=device  # Use selected computing device
)

# Bind the logger to the model
model.set_logger(logger)

# Periodically save the model to avoid loss in case of interruption
checkpoint_interval = args.total_timesteps // 10  # Save model at 10% progress intervals
for i in range(1, 11):
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
    model.save(f"{args.save_path}_checkpoint_{i}")

# Save the final trained model
model.save(args.save_path)
env.close()
