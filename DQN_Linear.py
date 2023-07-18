import math
import random
from collections import deque, namedtuple
from itertools import count
import yaml

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import save_scalars


# HyperParameters
with open('configs/DQN_Linear.yml') as f:
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

HIDDEN_LAYER1 = config['HIDDEN_LAYER1']
HIDDEN_LAYER2 = config['HIDDEN_LAYER2']
HIDDEN_LAYER3 = config['HIDDEN_LAYER3']

USE_CUDA = config['USE_CUDA']

env = gym.make('CartPole-v0', render_mode='human')
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], HIDDEN_LAYER1),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER1, HIDDEN_LAYER2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER2, HIDDEN_LAYER3),
            nn.ReLU(),
        )
        
        self.head = nn.Linear(HIDDEN_LAYER3, env.action_space.n)
        
    def forward(self, x):
        return self.head(self.net(x))

policy_net = DQN().to(device)
target_net = DQN().to(device)
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
    observation = env.reset()[0]
    observation = torch.tensor(np.expand_dims(observation, 0), device=device, dtype=torch.float32)
    for t in count():
        action = select_action(observation, stop_training)
        next_observation, reward, terminated, truncated, _ = env.step(action.item())
        cart_positions.append(next_observation[0])
        pole_angles.append(next_observation[2])
        done = terminated or truncated
        
        next_observation = torch.tensor(np.expand_dims(next_observation, 0), device=device, dtype=torch.float32)

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
        
        save_scalars('cart_positions', 'DQN_Linear', cart_positions)
        save_scalars('pole_angles', 'DQN_Linear', pole_angles)
        
        if done:
            episode_rewards.append(episode_reward)
            save_scalars('episode_rewards', 'DQN_Linear', episode_rewards, save_every=1)
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