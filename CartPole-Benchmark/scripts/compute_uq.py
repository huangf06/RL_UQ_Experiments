import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from utils.seed_manager import generate_seeds
from envs.make_env import make_cartpole

MODEL_DIR = "results/models"
RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_VERSIONS = [0,1,2,3]
NUM_AGENTS = 5
NUM_STATES = 100

def load_models(env_version):
    """Load trained PPO models for a given CartPole environment version."""
    SEEDS = generate_seeds(num_seeds=NUM_AGENTS)
    models = []
    for seed in SEEDS:
        model_path = os.path.join(MODEL_DIR, f"ppo_cartpole_{env_version}_seed{seed}.zip")
        if os.path.exists(model_path):
            models.append(PPO.load(model_path))
        else:
            print(f"Warning: Model {model_path} not found.")
    return models

def compute_policy_variance(obs, models):
    """Compute the variance of action probability distributions."""
    probabilities = []    
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  

    for model in models:
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            action_probs = dist.distribution.probs.cpu().numpy()
        probabilities.append(action_probs[0, 1])
    
    return np.var(probabilities)

uq_results = {}

for env_version in ENV_VERSIONS:
    env_name = f"CartPole-{env_version}"
    print(f"Processing UQ for {env_name}...")

    models = load_models(env_version)
    if not models:
        print(f"Skipping {env_name} as no models were found.")
        continue

    env = make_cartpole(env_version)
    obs, _ = env.reset()
    uq_values = []

    for _ in range(NUM_STATES):
        var_p1 = compute_policy_variance(obs, models)
        uq_values.append(var_p1)
        
        obs, _, done, _, _ = env.step(env.action_space.sample())  
        if done:
            obs, _ = env.reset()

    env.close()
    
    uq_results[env_name] = uq_values
    print(f"Completed UQ calculation for {env_name}")

uq_file = os.path.join(RESULTS_DIR, "uq_data.npy")
np.save(uq_file, uq_results)
print(f"UQ results saved to {uq_file}")
