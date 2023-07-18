from functools import partial

from utils import plot_fig


plot_sarsa_cart_positions = partial(plot_fig, kind='cart_positions', algo='SARSA')
plot_sarsa_pole_angles = partial(plot_fig, kind='pole_angles', algo='SARSA')
plot_sarsa_rewards = partial(plot_fig, kind='episode_rewards', algo='SARSA')

plot_q_learning_cart_positions = partial(plot_fig, kind='cart_positions', algo='Q_Learning')
plot_q_learning_pole_angles = partial(plot_fig, kind='pole_angles', algo='Q_Learning')
plot_q_learning_rewards = partial(plot_fig, kind='episode_rewards', algo='Q_Learning')

plot_dqn_linear_cart_positions = partial(plot_fig, kind='cart_positions', algo='DQN_Linear')
plot_dqn_linear_pole_angles = partial(plot_fig, kind='pole_angles', algo='DQN_Linear')
plot_dqn_linear_rewards = partial(plot_fig, kind='episode_rewards', algo='DQN_Linear')

plot_dqn_conv_cart_positions = partial(plot_fig, kind='cart_positions', algo='DQN_Conv')
plot_dqn_conv_pole_angles = partial(plot_fig, kind='pole_angles', algo='DQN_Conv')
plot_dqn_conv_rewards = partial(plot_fig, kind='episode_rewards', algo='DQN_Conv')


if __name__ == '__main__':
    plot_dqn_linear_pole_angles()