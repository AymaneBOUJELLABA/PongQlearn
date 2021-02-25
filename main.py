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
    def __init__(self):

        print('\nchoose a value:')
        type = input('1. agentRL vs human\n2. agentRL vs agentAI\n3. agentRl vs agentRL\nvalue = ')
        self.game = g.Game(type)
        self.games_played = 0

    def beginPlaying(self, episodes):
        self.game.play(episodes)


if __name__ == '__main__':
    gl = GameLearning()
    gl.beginPlaying(1000)
