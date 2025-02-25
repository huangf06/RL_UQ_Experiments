import gym
import torch
import imageio
import realworldrl_suite.gym_wrappers as wrappers

# Load environment
env = wrappers.RealWorldWrapper(gym.make("Walker-v0"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size).to(device)
policy_net.load_state_dict(torch.load("../models/walker_dqn.pth", map_location=device))
policy_net.eval()

def get_action(state):
    with torch.no_grad():
        state = torch.FloatTensor(state).to(device)
        return torch.argmax(policy_net(state)).item()

# Generate video
video_path = "../results/walker_dqn_run.mp4"
frames = []

state = env.reset()
done = False

for _ in range(500):
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    
    action = get_action(state)
    state, _, done, _ = env.step(action)
    
    if done:
        break

env.close()

# Save video
imageio.mimsave(video_path, frames, fps=30)
