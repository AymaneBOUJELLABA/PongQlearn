import game as g
import agent as ag
import matplotlib.pylab as plt
import numpy as np


def plot_agent_reward(rewards):
    """ Function to plot agent's accumulated reward vs. iteration """
    plt.plot(np.cumsum(rewards))
    plt.title('Agent Cumulative Reward vs. Iteration')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()


class GameLearning:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.01):
        self.agent = ag.Qlearning(alpha, gamma, epsilon)
        self.game = g.Game(self.agent)
        self.games_played = 0
        # plot_agent_reward(self.agent.reward)

    def beginPlaying(self, episodes):
        self.game.play(episodes)


if __name__ == '__main__':
    gl = GameLearning()
    gl.beginPlaying(1000)
