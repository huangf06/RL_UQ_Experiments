import os
import torch
import gymnasium as gym
import argparse
import cv2
import numpy as np
from stable_baselines3 import DQN

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/dqn_cartpole", help="Path to saved model")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes")
parser.add_argument("--device", type=str, default="auto", help="Choose device: auto, cpu, cuda")
parser.add_argument("--save_video", type=str, default="cartpole.mp4", help="Path to save the evaluation video")
args = parser.parse_args()

# Automatically select device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

# Load the trained model (use latest checkpoint if available)
if not os.path.exists(args.model_path + ".zip"):
    checkpoints = [f for f in os.listdir(os.path.dirname(args.model_path)) if "checkpoint" in f and f.endswith(".zip")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        args.model_path = os.path.join(os.path.dirname(args.model_path), latest_checkpoint)
        print(f"Using latest checkpoint: {args.model_path}")
    else:
        raise FileNotFoundError(f"No trained model found at {args.model_path}")

print(f"Loading model from: {args.model_path}")
model = DQN.load(args.model_path, device=device)

# Create environment with "rgb_array" mode for video saving
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Evaluate model and record video
total_rewards = []
frames = []

for episode in range(args.num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        frame = env.render()
        frames.append(frame)  # Save frame for video
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    total_rewards.append(episode_reward)
    print(f"Episode {episode+1}: Reward = {episode_reward}")

# Save evaluation results
results_path = "logs/evaluation_results.txt"
with open(results_path, "a") as f:
    f.write(f"Model: {args.model_path}, Episodes: {args.num_episodes}, Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}\n")

print(f"Evaluation complete. Results saved to {results_path}")

# Save video
if frames:
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    out.release()
    print(f"Video saved as {args.save_video}")

env.close()
