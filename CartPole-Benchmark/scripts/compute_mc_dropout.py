import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from envs.make_env import make_cartpole
import random

MODEL_DIR = "results/models"
RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_VERSIONS = [0, 1, 2, 3]  
NUM_STATES = 100  
MC_DROPOUT_SAMPLES = 20  

def find_model(env_version):
    """Select a random PPO model from the five trained seed models."""
    available_models = [
        f for f in os.listdir(MODEL_DIR) 
        if f.startswith(f"ppo_cartpole_{env_version}_seed") and f.endswith(".zip")
    ]
    
    if not available_models:
        print(f"Warning: No model found for CartPole-{env_version}")
        return None

    selected_model = random.choice(available_models)
    model_path = os.path.join(MODEL_DIR, selected_model)
    print(f"Using model {selected_model} for CartPole-{env_version}")
    return PPO.load(model_path)

def compute_mc_dropout_variance(obs, model, num_samples=MC_DROPOUT_SAMPLES):
    """Perform multiple stochastic forward passes to estimate uncertainty."""
    probabilities = []
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    model.policy.train()

    for _ in range(num_samples):
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            action_probs = dist.distribution.probs.cpu().numpy()
        probabilities.append(action_probs[0, 1])

    model.policy.eval()
    return np.var(probabilities)

uq_results = {}

for env_version in ENV_VERSIONS:
    env_name = f"CartPole-{env_version}"
    print(f"Processing MC Dropout UQ for {env_name}...")

    model = find_model(env_version)
    if not model:
        continue

    env = make_cartpole(env_version)
    obs, _ = env.reset()
    uq_values = []

    for _ in range(NUM_STATES):
        var_p1 = compute_mc_dropout_variance(obs, model, MC_DROPOUT_SAMPLES)
        uq_values.append(var_p1)

        obs, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            obs, _ = env.reset()

    env.close()
    
    uq_results[env_name] = uq_values
    print(f"Completed MC Dropout UQ calculation for {env_name}")

uq_file = os.path.join(RESULTS_DIR, "uq_mc_dropout.npy")
np.save(uq_file, uq_results)
print(f"UQ results (MC Dropout) saved to {uq_file}")
