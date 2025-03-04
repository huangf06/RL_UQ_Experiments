import gymnasium as gym
import imageio


env = gym.make("CartPole-v1", render_mode="rgb_array")

seed = 299409666585067956067793793503220504408
state, _ = env.reset(seed=seed)
# print("初始状态:", state, info)
# print(env.observation_space)
# print(env.action_space)
# print(env.np_random_seed)
# print(env.metadata)

frames = []

for step in range(100):  # 运行 10 步
    action = env.action_space.sample() # 让小车一直向右推
    next_state, reward, done, _, _ = env.step(action)  # 计算新状态
    print(f"Step {step}: Action={action}, Next State={next_state}")

    frames.append(env.render())
    
    if done:
        print("游戏结束！")
        break

env.close()

imageio.mimsave('cartpole.gif',frames, fps=30)