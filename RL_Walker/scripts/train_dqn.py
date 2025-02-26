import gymnasium as gym
import numpy as np
import cv2
import logging
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
log_file = "walker2d_training.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

### === 1. Train the Walker2d Agent === ###
# Choose RL algorithm (PPO or SAC)
USE_SAC = False  # Set to True to use SAC instead of PPO

# Create the environment
env = gym.make("Walker2d-v5")

# Select the model
if USE_SAC:
    model = SAC("MlpPolicy", env, verbose=1)
    model_name = "walker2d_sac"
else:
    model = PPO("MlpPolicy", env, verbose=1)
    model_name = "walker2d_ppo"

# Train the model for a sufficient number of timesteps
TOTAL_TIMESTEPS = 500_000  # Train for 500K steps to ensure the agent learns to walk
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save the trained model
model.save(model_name)
logging.info(f"Training completed. Model saved as {model_name}.zip")

# Close the environment
env.close()


### === 2. Evaluate the Walker2d Agent === ###
# Load the trained model
model = PPO.load(model_name) if not USE_SAC else SAC.load(model_name)

# Create evaluation environment
env = gym.make("Walker2d-v5")

# Evaluate over multiple episodes
n_episodes = 10
episode_rewards = []
episode_lengths = []

for _ in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

# Compute evaluation statistics
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
mean_length = np.mean(episode_lengths)

logging.info(f"Evaluation Results (over {n_episodes} episodes):")
logging.info(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
logging.info(f"Mean Episode Length: {mean_length:.2f} steps")
logging.info(f"Max Episode Reward: {np.max(episode_rewards):.2f}")
logging.info(f"Min Episode Reward: {np.min(episode_rewards):.2f}")

# Close the evaluation environment
env.close()


### === 3. Record Walker2d Performance Video === ###
# Create a new environment for rendering
env = gym.make("Walker2d-v5", render_mode="rgb_array")

# Record frames for the video
video_frames = []
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    # Capture the current frame
    frame = env.render()
    video_frames.append(frame)

# Close the environment
env.close()

# Save the video in the current directory
video_path = "walker2d_run.mp4"

# Write video frames to an MP4 file
height, width, _ = video_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

for frame in video_frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()

logging.info(f"Walker2d performance video recorded. Saved as {video_path}")
