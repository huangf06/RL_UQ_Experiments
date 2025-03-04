import gymnasium as gym
import os
import logging
import imageio
import numpy as np
from stable_baselines3 import PPO

# Configure logging
log_file = "walker2d_eval.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

# Load the trained model
model_path = "walker2d_ppo.zip"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = PPO.load(model_path)

# Set video output path
video_dir = "results"
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, "walker2d_eval.mp4")

# Create the environment
env = gym.make("Walker2d-v5", render_mode="rgb_array")

# Record frames
video_frames = []
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    frame = env.render()
    video_frames.append(frame)

# Save video using imageio
imageio.mimsave(video_path, video_frames, fps=30)

# Close environment
env.close()

# Log success message
logging.info(f"Walker2d evaluation video saved at {video_path}")
print(f"✅ Video successfully saved at: {video_path}")
