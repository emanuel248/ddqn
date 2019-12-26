import torch.nn as nn
import torch.nn.functional as F

class Q_Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Q_Net, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        h = self.relu1(self.fc1(x))
        h = self.relu2(self.fc2(h))
        y = self.fc3(h)
        return y

    def reset(self):
        self.zero_grad()