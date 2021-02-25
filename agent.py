import numpy as np
import random


class Qlearning:

    # env = [height ?]
    #
    # actions = [0,1,2,3,4,5,...]
    # states = [0,1,2,3,4,5,....] "depending on page/bar heights

    def __init__(self, alpha, gamma, eps, pageHeight, barHeight, eps_decay=0.):
        # initialise number of states
        self.states = int(pageHeight / barHeight)
        print("number of states : ", self.states)
        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        # Initialise Q-Table
        self.qTable = np.matrix(np.zeros(shape=[self.states, self.states]))
        self.rewards = []

        print("Qtable : ", self.qTable.shape)

    def getReward(self, ball, bar):
        if bar.top <= ball.centery <= bar.bottom:
            return 1
        else:
            return -1

    def get_state_from_pos(self, y, barHeight, pageHeight):
        a = 0
        b = barHeight
        for i in range(0, pageHeight, barHeight):
            if a <= y <= b:
                return int((b / barHeight) - 1)
            else:
                a += barHeight
                b += barHeight

    def get_action(self, s):
        print("get action for : " , s)
        if s < 0:
            s = 0
        if s >= self.states:
            s = self.states -1
        a = np.argmax(self.qTable[s, :])

        if a < 0:
            a = 0
        if a > self.states:
            a = self.states
        print(" action chosen : ", a)
        print(" from state : ", s)
        return a

    def update(self, s, bar, ball):
        """
        Perform the Q-Learning update of Q values.
        Parameters
        ----------
        ball : ball
               object for calculating position
        bar : the height of the bar
        s : string
            previous state
        s_ : string
            new state
        a : action
        """
        if s < 0:
            s = 0
        if s >= self.states:
            s = self.states - 1

        exp_threshold = random.uniform(0,1)
        if exp_threshold > self.eps:
            a = self.get_action(s)
            print("exploiting : ",a)
        else:
            a = random.choice([i for i in range(self.states)])
            print("exploring : ",a)

        if a >= 0:
            s_ = a

        reward = self.getReward(ball, bar)
        self.rewards.append(reward)

        self.qTable[s, a] += self.alpha * (reward + self.gamma * np.max(self.qTable[s_, :]) - self.qTable[s, a])

        return s_ * bar.height
