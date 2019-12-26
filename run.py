from environment import QEnvironment
import pandas as pd
import numpy as np
from train_ddqn import train_ddqn
from train_dqn import train_dqn
import argparse
import locale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    #data = pd.read_csv('Bitfinex_ETHUSD_1h.csv', skiprows=1)
    data = pd.read_csv('tsla.us.txt')
    data['Date'] = pd.to_datetime(data['Date'])
    #locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    #data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %I-%p', errors='coerce')
    data = data.set_index('Date', drop=False)
    print(data.index.min(), data.index.max())
    data.head()

    date_split = '2016-10-18'
    train = data[:date_split]
    test = data[date_split:]
    print(f'train {len(train)} | test {len(test)}') 

    # just a short test
    env = QEnvironment(train, history_t=90)
    #print(env.reset())
    for _ in range(3):
        pact = np.random.randint(3)
        #print(env.step(pact))

    Q, total_losses, total_rewards = train_ddqn(QEnvironment(train))
    #Q, total_losses, total_rewards = train_dqn(QEnvironment(train))
#    print(f'total loss: {total_losses}, total_rewards: {total_rewards}')
