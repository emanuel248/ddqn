from qnet import Q_Net, Q_Net_RNN
from environment import QEnvironment
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
import string
import locale

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
from utils import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Bitfinex_ETHUSD_1h.csv', help='*.csv file path')
    parser.add_argument('--weights', type=str, help='*.weights file path')
    opt = parser.parse_args()
    
    if opt.weights is None:
        print('weights file required')
        sys.exit()
        

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'using {device}')

    test,train = load_data(opt.data)
    env = QEnvironment(test, hist_t=20)
    #find out input size
    _inp = env.reset()
    actions = ['nop', 'buy', 'sell']
    Q = Q_Net_RNN(env, device, input_size=len(_inp), output_size=len(actions))
    Q.load_state_dict(torch.load(opt.weights))
    Q.to(device)

    step_max = env.len()
    prev_obs = env.reset()
    total_reward = 0

    print(f'going through {step_max} steps')
    for _ in range(step_max):
        prev_act, act_val = Q(torch.tensor(prev_obs, dtype=torch.float32).reshape(1,-1).to(device))
        prev_act = torch.argmax(prev_act).item()

        # act
        print(prev_act)
        obs, reward, done = env.step(prev_act)

        # next step
        total_reward += reward
        prev_obs = obs

    print(f'reward: {total_reward}, balance: {env.balance}')
    actions = ['nop','buy','sell']
    categories = [actions[int(i)] for i in env.tx_log[:,2]]
    #plot test run
    plt.figure(figsize=(15,8))
    sns.lineplot(x=env.tx_log[:,0], y=env.tx_log[:,1], linewidth=0.8)
    sns.scatterplot(x=env.tx_log[:,0], y=env.tx_log[:,1], hue=categories, s=20)
    plt.show(block=True)