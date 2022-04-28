import torch.nn as nn
import torch.nn.functional as F
import torch

Label = {"rock": 0, "scissors": 1, "pager": 2}

def getLable(name):
    return Label[name]

def getLabelName(index):
    for name in Label:
        if Label[name]==index:
            return name

def getWin(a, b):
    if a == b:
        return 0
    if a == 0 and b == 1:
        return 1
    if a == 1 and b == 2:
        return 1
    if a == 2 and b == 0:
        return 1
    return -1

def getWinAction(index):
    if index==0:
        return 2
    if index==1:
        return 0
    if index==2:
        return 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out 
