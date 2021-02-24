import numpy as np

import random


def getActionIndex(action):
    switcher = {
        "UP": 0,
        "DOWN": 1,
    }
    return switcher.get(action, -1)


def getStateIndex(state):
    switcher = {
        "WIN": 0,
        "LOOSE": 1,
        "TOUCHBALL": 2,
        "BOTTOM":4,
        "TOP":5
    }
    return switcher.get(state, 3)


class Qlearning:

    # env = [height ?]
    #
    # actions = ["UP", "DOWN"]
    #
    # States = ["WIN", "LOOSE","TOUCH BALL","DEFAULT","AGENT IS ON BOTTOM", "AGENT IS ON TOP]
    # Rewards = ["1" , "-1"   , "0.1"       , "0",    ["UP" : 0.05,"DOWN": -0.05],
    #                                                                       ["UP": -0.05, "DOWN": 0.05 ] ]
    reward = np.matrix('1 1; -1 -1; 0.1 0.1; 0 0; 0.05 -0.05; -0.05 0.05')

    def __init__(self, alpha, gamma, eps, eps_decay=0.):
        #
        self.gameState = "DEFAULT"
        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        # Initialise Q-Table
        self.qTable = np.matrix(np.zeros(shape=[6, 2]))

    def setState(self, state):
        self.state = state

    def get_action(self, s):
        print("CURRENT STATE IS : " + s)
        idx = getStateIndex(s)
        a, b = np.where(self.qTable == np.max(self.qTable[idx]))
        print("a : " , a)
        print("b : ", b)

        if b.any() == 0:
            AgentAction = "UP"
        else:
            AgentAction = "DOWN"
        if s == "BOTTOM":
            AgentAction = "UP"
        elif s == "TOP":
            AgentAction = "DOWN"
        print("agent will go : ", AgentAction)
        return AgentAction

    def update(self, s, s_, a):
        """
        Perform the Q-Learning update of Q values.
        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        """
        sIdx = getStateIndex(s)
        aIdx = getActionIndex(a)
        newSIdx = getStateIndex(s_)
        self.qTable[sIdx, aIdx] = self.reward[sIdx, aIdx] + self.gamma * np.max(self.qTable[newSIdx])
        # Q(s,a) = R(s,a) + self.gamma * np.max(Q(s_, all_actions()))

    def train_agent(self, episodes):
        for i in range(episodes):
            curr_action = random.choice(["UP","DOWN"])
            curr_state = random.choice(["WIN", "LOOSE", "TOUCHBALL", "DEFAULT","BOTTOM","TOP"])
            curr_nextstate = random.choice(["WIN", "LOOSE", "TOUCHBALL", "DEFAULT","BOTTOM","TOP"])
            self.update(curr_state, curr_nextstate, curr_action)
