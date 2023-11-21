import numpy as np
import random

def thompson_sampling(trials, wins, p):
    pbeta = [0, 0, 0, 0, 0]
    for i in range(0, len(trials)):
        pbeta[i] = np.random.beta(wins[i]+1, trials[i]-wins[i]+1)

    choice = np.argmax(pbeta)

    trials[choice] += 1
    if p[choice] > random.random():
        wins[choice] += 1


def test():
    p = [0.1, 0.2, 0.3, 0.4, 0.5]
    trials = np.array([0, 0, 0, 0, 0])
    wins = np.array([0, 0, 0, 0, 0])
    for i in range(0, 10000):
        thompson_sampling(trials, wins, p)

    print(trials)
    print(wins)
    print(wins/trials)


test()
