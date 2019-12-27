import torch.nn as nn
import torch.nn.functional as F

class Q_Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, droprate=0.2):
        super(Q_Net, self).__init__()

        self.output_size = output_size
        self.dilated_conv1 = nn.Conv1d(1, 64, kernel_size=2, dilation=2)
        self.dilated_conv2 = nn.Conv1d(64, 64, kernel_size=1, dilation=2)

        self.fc1_adv = nn.Linear(64*90, 512)
        self.fc1_val = nn.Linear(64*90, 512)

        self.fc2_adv = nn.Linear(512, output_size)
        self.fc2_val = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        y = self.relu(self.dilated_conv1(x))
        y = self.relu(self.dilated_conv2(y))
        #flatten to linear by batch size
        feature_vec = y.view(x.size(0), -1)
        adv = self.dropout(self.relu(self.fc1_adv(feature_vec)))
        val = self.dropout(self.relu(self.fc1_val(feature_vec)))

        adv = self.dropout(self.fc2_adv(adv))
        val = self.dropout(self.fc2_val(val).expand(feature_vec.size(0), self.output_size))
        
        y = val + adv - adv.mean(1).unsqueeze(1).expand(feature_vec.size(0), self.output_size)
        return y

    def reset(self):
        self.zero_grad()
