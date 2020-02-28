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
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (2, 20)
latent_dim = (2, 128)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, num_features=2, hidden_size=256, dilation=2):
        super(Generator, self).__init__()

        self.dilation = dilation
        self.hidden_size = hidden_size
        self.num_features = num_features

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
            nn.Linear(int(126*64), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 40)
        )

    def forward(self, z):
        x = self.features(z)
        x = x.view(z.shape[0], -1)
        x = self.classifier(x)
        return x.view(z.shape[0], self.num_features, -1)


class Discriminator(nn.Module):
    def __init__(self, num_features=2, hidden_size=256, dilation=2):
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
            nn.Linear(int(18*64), 512),
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
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

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
dataset = CsvDataset(data[['Open','Close']], 20) # 20 entries, 2 features

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

allchar = string.ascii_letters + string.digits
proj_name = "".join(choice(allchar) for x in range(6))

writer = SummaryWriter(f'gan_logs/gan_{proj_name}', flush_secs=20)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
best_loss_D = 0.3
best_loss_G = 0.3
for epoch in tqdm(range(opt.epochs), ncols=100, ascii=True):

    train_loss_D = 0.0
    train_loss_G = 0.0
    for i, imgs in enumerate(train_loader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], *latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        train_loss_D += loss_D.item()

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            train_loss_G += loss_G.item()

            loss_G.backward()
            optimizer_G.step()

            #print(
            #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #    % (epoch, opt.epochs, batches_done % len(train_loader), len(train_loader), loss_D.item(), loss_G.item())
            #)

        batches_done += 1
    writer.add_scalar('Discriminator', train_loss_D/len(train_loader), epoch)
    writer.add_scalar('Generator', train_loss_G/len(train_loader)/opt.n_critic, epoch)

    if epoch > 20 and epoch % 10 == 0:
        loss_D_val = torch.tensor([0.0])
        loss_G_val = torch.tensor([0.0])
        for i, data_ in enumerate(validation_loader):
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], *latent_dim))))
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D_val += -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            gen_imgs = generator(z).detach()
            # Adversarial loss
            loss_G_val += -torch.mean(discriminator(gen_imgs))
        writer.add_scalar('Validation D', loss_D_val.item()/len(validation_loader), epoch)
        writer.add_scalar('Validation G', loss_G_val.item()/len(validation_loader), epoch)
        if best_loss_D > (loss_D_val.item()/len(validation_loader)) and -best_loss_D < (loss_D_val.item()/len(validation_loader)):
            torch.save(discriminator,'model_D.pt')
            best_loss_D = loss_D_val.item()/len(validation_loader)
        if best_loss_G > (loss_G_val.item()/len(validation_loader)) and -best_loss_G < (loss_G_val.item()/len(validation_loader)):
            torch.save(generator,'model_G.pt')
            best_loss_G = loss_G_val.item()/len(validation_loader)