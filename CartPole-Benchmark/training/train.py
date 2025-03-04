import os
import argparse
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from envs.make_env import make_cartpole

CONFIG_PATH = "training/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

os.makedirs(config["experiment"]["log_dir"], exist_ok=True)
os.makedirs(config["experiment"]["model_dir"], exist_ok=True)

def train(env_version=None, total_timesteps=None):
    """
    Train a PPO agent and record training data.
    
    Args: 
        env_version (int): CartPole environment variant (0-3).
        total_timesteps (int): training timesteps.
    """

    if env_version is None:
        env_version = config["experiment"]['env_version']
    if total_timesteps is None:
        total_timesteps = config["training"]["total_timesteps"]
    
    env = make_cartpole(env_version)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        verbose=1,
        tensorboard_log=config["experiment"]["log_dir"]
    )

    logger = configure(config["experiment"]["log_dir"], ["stdout", "tensorboard"])
    model.set_logger(logger)

    print(f"Training PPO on CartPole-{env_version} for {total_timesteps} steps.")
    model.learn(total_timesteps=total_timesteps)

    model_path = os.path.join(config["experiment"]["model_dir"], f"ppo_cartpole_{env_version}.zip")
    model.save(model_path)
    print(f"Finished training. Model saved to {model_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_version", type=int, default=None, help="CartPole Variant (0-3)")
    parser.add_argument("--timesteps", type=int, default=None, help="Time Steps")
    args = parser.parse_args()

    train(args.env_version, args.timesteps)
