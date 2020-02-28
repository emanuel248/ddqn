import argparse
import os
import numpy as np
import math
import sys

from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import string
from random import randint,choice
from sklearn.externals import joblib

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=3e-6, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--generate_only", type=bool, default=False, help="Generate samples")
opt = parser.parse_args()
print(opt)

img_shape = (2, 20)
latent_dim = (2, 128)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, num_features=2, hidden_size=128, dilation=2,steps=20):
        super(Generator, self).__init__()

        self.dilation = dilation
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.steps = steps

        self.features = nn.Sequential(
            
            # First Layer
            # Input
            nn.Conv1d(num_features, hidden_size, kernel_size=2, dilation=self.dilation),
            nn.LeakyReLU(),

            # Layer 2
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, dilation=self.dilation),
            nn.LeakyReLU(),

            # Layer 3
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation),
            nn.LeakyReLU(),

            # Layer 4
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation),
            nn.LeakyReLU(),
            
             # Output layer
            nn.Conv1d(hidden_size, 64, kernel_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(126*64), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.steps*self.num_features)
        )

    def forward(self, z):
        x = self.features(z)
        x = x.view(z.shape[0], -1)
        x = self.classifier(x)
        return x.view(z.shape[0], self.num_features, -1)


class Discriminator(nn.Module):
    def __init__(self, num_features=2, hidden_size=128, dilation=2,steps=20):
        super(Discriminator, self).__init__()

        self.dilation = dilation
        self.features = nn.Sequential(
            
            # First Layer
            # Input
            nn.Conv1d(num_features, hidden_size, kernel_size=2, dilation=self.dilation),
            nn.LeakyReLU(),

            # Layer 2
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation),
            nn.LeakyReLU(),

            # Layer 3
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation),
            nn.LeakyReLU(),

            # Layer 4
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation),
            nn.LeakyReLU(),
            
             # Output layer
            nn.Conv1d(hidden_size, 64, kernel_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int((self.steps-2)*64), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        x = self.features(img)
        x = x.view(img.shape[0], -1)
        validity = self.classifier(x)
        return validity


# Initialize generator and discriminator
generator = torch.load('model_G.pt', map_location=torch.device('cpu')).double()
discriminator = torch.load('model_D.pt', map_location=torch.device('cpu')).double()


# Configure data loader
class CsvDataset(Dataset):
    def __init__(self, data_frame, q):
        self.scaler = MinMaxScaler()
        self.scaler.fit(data_frame.values)
        self.data = self.scaler.transform(data_frame.values)
        self.q = q

    def __len__(self):
        return int(np.ceil(self.data.shape[0]//self.q))

    def __getitem__(self, index):
        data = self.data[index*self.q:(index+1)*self.q]
        return np.transpose(data,(1,0))

validation_split = .2
data = pd.read_csv('EURUSD_15m_BID_01.01.2010-31.12.2016.csv', skiprows=0)
dataset = CsvDataset(data[['Open','Close']], img_shape[1]) # 20 entries, 2 features

joblib.dump(dataset.scaler, 'minmax_scaler.pkl')

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, opt.batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                sampler=valid_sampler)

# ----------
#  Validation
# ----------
if opt.generate_only==False:
    loss_D_val = torch.tensor([0.0])
    loss_G_val = torch.tensor([0.0])
    for i, real_imgs in tqdm(enumerate(validation_loader), ncols=100, ascii=True):
        # Sample noise as generator input
        z = torch.tensor(np.random.normal(0, 1, (real_imgs.shape[0], *latent_dim)))
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D_val += -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        gen_imgs = generator(z).detach()
        # Adversarial loss
        loss_G_val += -torch.mean(discriminator(gen_imgs))
    print('Validation D', loss_D_val.item()/len(validation_loader))
    print('Validation G', loss_G_val.item()/len(validation_loader))

for _ in range(3):
    z = torch.tensor(np.random.normal(0, 1, (1, *latent_dim)))
    fake_imgs = generator(z).detach()
    print(fake_imgs.shape)
    fake_data = dataset.scaler.inverse_transform(fake_imgs.squeeze(0).permute(1,0).numpy())
    print(fake_data)