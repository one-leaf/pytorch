import gym
import math
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
import os

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0
        
    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 =  torch.tensor(s0, dtype=torch.float).view(1,-1)
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    def put(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return
        
        samples = random.sample( self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor( s0, dtype=torch.float)
        a0 = torch.tensor( a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor( r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor( s1, dtype=torch.float)
        
        y_true = r1 + self.gamma * torch.max( self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

def plot(score, mean):
    
    plt.figure(2)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean)-1, mean[-1], str(mean[-1]))
    plt.pause(0.001) 

if __name__ == '__main__':

    modle_file = "data/save/15_checkpoint.tar"

    env = gym.make('CartPole-v0').unwrapped

    params = {
        'gamma': 0.85,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200, 
        'lr': 1e-3,
        'capacity': 100000,
        'batch_size': 256,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n   
    }
    agent = Agent(**params)

    if os.path.exists(modle_file):
        agent.eval_net.load_state_dict(torch.load(modle_file)["eval_net"])

    score = []
    mean = []

    avg_reward = 0.1
    for episode in range(100000):
        s0 = env.reset()
        for t in count():
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            
            if not done:
                r1 = -0.1
            else:
                r1 = t # math.exp(-1. * (t+1) / avg_reward )

            agent.put(s0, a0, r1, s1)
            
            if done:
                break

            s0 = s1
            loss = agent.learn()
            
        # score.append(t)
        # mean.append( sum(score[-100:])/100)
        avg_reward = avg_reward*0.99 + t*0.01
        # plot(score, mean)

        if episode % 10==0:
            if loss!=None:
                print(episode, t,"/", avg_reward, "loss:", loss.item()) 
            torch.save({    'eval_net': agent.eval_net.state_dict(),
                }, modle_file)