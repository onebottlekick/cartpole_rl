import os
import pickle

import matplotlib.pyplot as plt
import torch


def save_scalars(kind:str, algo:str, target:list, save_every:int=3000) -> None:
    save_path = f'scalars/{kind}/{algo}'
    os.makedirs(save_path, exist_ok=True)
    if len(target) % save_every == 0:
        with open(f'{save_path}/{str(len(target))}.pkl', 'wb') as f:
            pickle.dump(target, f)
    with open(f'scalars/{kind}/{algo}/final.pkl', 'wb') as f:
        pickle.dump(target, f)
        
        
def plot_fig(kind:str, algo:str, save_fig:bool=False) -> None:
    scalar_path = f'scalars/{kind}/{algo}'
    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)
    with open(f'{scalar_path}/final.pkl', 'rb') as f:
        target = pickle.load(f)
        
    if kind == 'episode_rewards':
        rewards_t = torch.tensor(target, dtype=torch.float)
        num_episodes = len(rewards_t)
        plt.title(f'{algo} - Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy(), label='Score')
        if num_episodes >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label='Last 100 Mean')
        plt.legend(loc='upper left')
    elif kind == 'cart_positions':
        plt.plot(target)
        plt.xlabel('Steps')
        plt.ylabel('Cart Position')
    elif kind == 'pole_angles':
        plt.plot(target)
        plt.xlabel('Steps')
        plt.ylabel('Pole Angle')
        
    if save_fig:
        plt.savefig(f'{save_path}/{kind}_{algo}.png')
        
    plt.show()