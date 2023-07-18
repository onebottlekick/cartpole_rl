import math
import yaml

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import save_scalars


env = gym.make('CartPole-v0', render_mode='human')

# HyperParameters
with open('configs/Q_Learning.yml') as f:
    config = yaml.safe_load(f)
    
NUM_EPISODES = config['NUM_EPISODES']
GAMMA = config['GAMMA']
ALPHA = config['ALPHA']
EPSILON = config['EPSILON']

theta_minmax = env.observation_space.high[2]
theta_dot_minmax = math.radians(50)
theta_state_size = 50
theta_dot_state_size = 50
Q_TABLE = np.random.randn(theta_state_size, theta_dot_state_size, env.action_space.n)

def discretised_state(state):
    discrete_state = np.array([0, 0])
    theta_window = (theta_minmax - (-theta_minmax))/theta_state_size
    discrete_state[0] = (state[2] - (-theta_minmax))//theta_window
    discrete_state[0] = min(theta_state_size - 1, max(0, discrete_state[0]))
    
    theta_dot_window = (theta_dot_minmax - (-theta_dot_minmax))/theta_dot_state_size
    discrete_state[1] = (state[3] - (-theta_dot_minmax))//theta_dot_window
    discrete_state[1] = min(theta_dot_state_size - 1, max(0, discrete_state[1]))

    return tuple(discrete_state.astype(int))


episode_rewards = []
cart_positions = []
pole_angles = []

for episode in range(NUM_EPISODES):
    episode_reward = 0
    done = False
    
    curr_discrete_observation = discretised_state(env.reset()[0])
    
    if np.random.random() > EPSILON:
        action = np.argmax(Q_TABLE[curr_discrete_observation])
    else:
        action = env.action_space.sample()
        
    while not done:
        new_observation, reward, terminated, truncated, _  = env.step(action)
        cart_positions.append(new_observation[0])
        pole_angles.append(new_observation[2])
        done = terminated or truncated
        new_discrete_observation = discretised_state(new_observation)

        if np.random.random() > EPSILON:
            new_action = np.argmax(Q_TABLE[new_discrete_observation])
        else:
            new_action = env.action_space.sample()
            
        if not done:
            curr_q = Q_TABLE[curr_discrete_observation + (action, )]
            max_future_q = np.max(Q_TABLE[new_discrete_observation + (new_action, )])
            new_q = curr_q + ALPHA*(reward + GAMMA*max_future_q - curr_q)
            Q_TABLE[curr_discrete_observation + (action, )] = new_q
            
            save_scalars('cart_positions', 'Q_Learing', cart_positions)
            save_scalars('pole_angles', 'Q_Learing', pole_angles)

        curr_discrete_observation = new_discrete_observation
        action = new_action
        
        episode_reward += reward
    
    episode_rewards.append(episode_reward)
    save_scalars('episode_rewards', 'Q_Learing', episode_rewards, save_every=1)