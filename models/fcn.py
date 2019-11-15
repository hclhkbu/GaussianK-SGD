# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN5Net(nn.Module):
    def __init__(self, sparse=False, sparsity=None):
        super(FCN5Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.name = 'fcn5net'

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.name = 'lr'

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        return x

