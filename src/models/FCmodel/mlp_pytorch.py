import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        self.fc1 = nn.Linear(128, 100)
#         self.fc1 = nn.Linear(128, 32)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(32, 32)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32,100)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        self.fc3_bn = nn.BatchNorm1d(100)

        self.fc4 = nn.Linear(100, 2)
        torch.nn.init.xavier_normal_(self.fc4.weight)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(-1,128)

        x = F.relu(self.fc1_bn(self.fc1(x)))

        x=self.drop(x)

        x = self.fc4(x)
        x = F.log_softmax(x,dim=1)
        return x
