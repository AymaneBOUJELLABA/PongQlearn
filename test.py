import game
import agent as QlAgent
import random
import numpy as np

agent = QlAgent.Qlearning(alpha=0.5, gamma=0.9, eps=0.01)
game = game.Game

print("qtables init ", agent.qTable)
print(" rewards init ", agent.reward)


def getStateIndex(state):
    switcher = {
        "WIN": 0,
        "LOOSE": 1,
        "TOUCHBALL": 2,
    }
    return switcher.get(state, 3)


print("WIN : ", getStateIndex("WIN"))
print("LOOSE : ", getStateIndex("LOOSE"))
print("TOUCHBALL : ", getStateIndex("TOUCHBALL"))
print("DEFAULT : ", getStateIndex("DEFAULT"))

print("QTABLE BEFORE : ", agent.qTable)
agent.train_agent(1000)
print("QTABLE AFTER : ", agent.qTable)
print("if action is DEFAULT agent will : " ,agent.get_action("BOTTOM"))

