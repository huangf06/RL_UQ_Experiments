import time
import argparse
from stable_baselines3 import PPO
from envs.make_env import make_cartpole

def test_agent(env_version=0, model_path="results/models/ppo_cartpole_0.zip", episodes=10, render=True):
    """
    Test a trained PPO agent in the CartPole environment.

    Args:
        env_version (int): CartPole variant (0-3).
        model_path (str): Path to the trained PPO model.
        episodes (int): Number of episodes to run.
        render (bool): Whether to render the environment.
    """
    env = make_cartpole(env_version, render_mode="human" if render else None)
    model = PPO.load(model_path)
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1

            if render:
                time.sleep(0.02)  # Control rendering speed

        print(f"Episode {episode+1}: Total Reward = {total_reward} | Steps = {step}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_version", type=int, default=0, help="CartPole variant (0-3)")
    parser.add_argument("--model_path", type=str, default="results/models/ppo_cartpole_0.zip", help="Path to PPO model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering for faster testing")

    args = parser.parse_args()
    
    test_agent(
        env_version=args.env_version, 
        model_path=args.model_path, 
        episodes=args.episodes, 
        render=not args.no_render
    )
