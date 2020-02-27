import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Q_Net(nn.Module):

    def __init__(self, input_size, output_size, drop_rate=0.5):
        super(Q_Net, self).__init__()

        self.output_size = output_size
        self.dilated_conv1 = nn.Conv1d(6, 64, kernel_size=2, dilation=2)
        self.dilated_conv2 = nn.Conv1d(64, 128, kernel_size=1, dilation=2)
        self.dilated_conv3 = nn.Conv1d(128, 128, kernel_size=1, dilation=2)
        self.batch_normalize = nn.BatchNorm1d(128)

        self.fc1_adv = nn.Linear(128*(input_size-2), 256)
        self.fc1_val = nn.Linear(128*(input_size-2), 256)

        self.fc2_adv = nn.Linear(256, output_size)
        self.fc2_val = nn.Linear(256, 1)
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        y = self.relu(self.dilated_conv1(x))
        y = self.relu(self.dilated_conv2(y))
        y = self.relu(self.dilated_conv3(y))
        y = self.batch_normalize(y)
        #flatten to linear by batch size
        feature_vec = y.view(x.size(0), -1)
        adv = self.dropout(self.relu(self.fc1_adv(feature_vec)))
        val = self.dropout(self.relu(self.fc1_val(feature_vec)))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(feature_vec.size(0), self.output_size)   
        
        y = val + adv - adv.mean(1).unsqueeze(1).expand(feature_vec.size(0), self.output_size)
        return y

    def reset(self):
        self.zero_grad()

class Q_Net_RNN(nn.Module):
    def __init__(self, device, input_size, output_size=4, drop_rate=0.5):
        super(Q_Net_RNN, self).__init__()

        self.device_ = device
        self.n_layers = 8
        self.hidden_dim = 128
        self.input_size = input_size

        self.input_layer = nn.Linear(input_size, 128)
        self.hidden_1 = nn.Linear(128, 128)
        self.hidden_2 = nn.Linear(self.hidden_dim, 128)
        self.rnn = nn.GRU(128, self.hidden_dim, self.n_layers)

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.saved_actions = []
        self.rewards = []

    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device_)
        
    def forward(self, x):
        batch_size = x.size(0)
        self.init_hidden(batch_size)

        x = x.view(batch_size, -1)
        x = torch.sigmoid(self.input_layer(x))
        x = torch.tanh(self.hidden_1(x))
        x, self.hidden_state = self.rnn(x.view(1,-1,128), self.hidden_state.data)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = F.relu(self.hidden_2(x))

        advantages = self.advantage_stream(x)
        values = self.value_stream(x)

        qvals = values + (advantages - advantages.mean())
        return qvals

    def reset(self):
        self.zero_grad()