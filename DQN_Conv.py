import math
import random
from collections import deque, namedtuple
from itertools import count
from PIL import Image
import yaml

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from utils import save_scalars


# HyperParameters
with open('configs/DQN_Conv.yml') as f:
    config = yaml.safe_load(f)
    
BATCH_SIZE = config['BATCH_SIZE']
GAMMA = config['GAMMA']
EPS_START = config['EPS_START']
EPS_END = config['EPS_END']
EPS_DECAY = config['EPS_DECAY']
TARGET_UPDATE = config['TARGET_UPDATE']
MEMORY_SIZE = config['MEMORY_SIZE']
END_SCORE = config['END_SCORE']
TRANING_STOP = config['TRAINING_STOP']
NUM_EPISODES = config['NUM_EPISODES']
LAST_EPISODES_NUM = config['LAST_EPISODES_NUM']

FRAMES = config['FRAMES']
RESIZE_PIXELS = config['RESIZE_PIXELS']
GRAYSCALE = config['GRAYSCALE']

HIDDEN_LAYER1 = config['HIDDEN_LAYER1']
HIDDEN_LAYER2 = config['HIDDEN_LAYER2']
HIDDEN_LAYER3 = config['HIDDEN_LAYER3']
KERNEL_SIZE = config['KERNEL_SIZE']
STRIDE = config['STRIDE']

USE_CUDA = config['USE_CUDA']

if GRAYSCALE == 0:
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
        T.ToTensor()
    ])
    nn_inputs = 3*FRAMES
    
else:
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
        T.Grayscale(),
        T.ToTensor()
    ])
    nn_inputs = FRAMES

env = gym.make('CartPole-v0', render_mode='rgb_array')
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nn_inputs, HIDDEN_LAYER1, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_LAYER1, HIDDEN_LAYER2, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER2),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_LAYER2, HIDDEN_LAYER3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER3),
            nn.ReLU(),
            nn.Flatten()
        )
        
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        linear_input_size = convh*convw*HIDDEN_LAYER3        
        self.head = nn.Linear(linear_input_size, env.action_space.n)
    
    # @staticmethod
    def conv2d_size_out(self, size, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=0):
        return (size + 2*padding - kernel_size)//stride + 1
        
    def forward(self, x):
        return self.head(self.net(x))
    
def get_cart_location(screen_width):
    world_width = env.x_threshold*2
    scale = screen_width/world_width
    return int(env.state[0]*scale + screen_width/2.0)

def get_screen():
    env.reset()
    screen = env.render().transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height*0.8)]
    view_width = int(screen_width*0.6)
    cart_location = get_cart_location(screen_width)
    
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width//2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width//2, cart_location + view_width//2)
        
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.tensor(screen)
    
    return resize(screen).unsqueeze(0).to(device)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

memory = ReplayMemory(MEMORY_SIZE)

def select_action(state, stop_training):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1.*steps_done/EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_observation)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_observation if s is not None])
    
    observation_batch = torch.cat(batch.observation)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(observation_batch).gather(1, action_batch)
    
    next_observation_values = torch.zeros(BATCH_SIZE, device=device)
    next_observation_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = reward_batch + GAMMA*next_observation_values
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
mean_last = deque([0]*LAST_EPISODES_NUM, LAST_EPISODES_NUM)

stop_training = False
steps_done = 0

episode_rewards = []
cart_positions = []
pole_angles = []
for episode in range(NUM_EPISODES):
    episode_reward = 0.0
    env.reset()
    init_screen = get_screen()
    screens = deque([init_screen]*FRAMES, FRAMES)
    observation = torch.cat(list(screens), dim=1)
    for t in count():
        action = select_action(observation, stop_training)
        _observation, reward, terminated, truncated, _ = env.step(action.item())
        cart_positions.append(_observation[0])
        pole_angles.append(_observation[2])
        done = terminated or truncated
        
        screens.append(get_screen())
        next_observation = torch.cat(list(screens), dim=1) if not done else None

        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        if t >= END_SCORE - 1:
            reward = reward + 20
            done = 1
        else:
            if done:
                reward = reward - 20
                
        memory.push(observation, action, next_observation, reward)
        
        observation = next_observation
        
        save_scalars('cart_positions', 'DQN_Conv', cart_positions)
        save_scalars('pole_angles', 'DQN_Conv', pole_angles)
        
        if done:
            episode_rewards.append(episode_reward)
            save_scalars('episode_rewards', 'DQN_Conv', episode_rewards, save_every=1)
            mean_last.append(t + 1)
            mean = 0
            
            for i in range(LAST_EPISODES_NUM):
                mean += mean_last[i]
            mean = mean / LAST_EPISODES_NUM
            if mean < TRANING_STOP and stop_training == False:
                optimize_model()
            else:
                stop_training = 1
            
            episode_reward = 0.0
            break
        
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())