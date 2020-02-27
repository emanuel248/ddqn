from environment import QEnvironment
import pandas as pd
import numpy as np
from train_ddqn import train_ddqn
from train_dqn import train_dqn
import argparse
import locale
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    data = pd.read_csv('Bitfinex_ETHUSD_1h.csv', skiprows=1)
    #data = pd.read_csv('tsla.us.txt')
    #data['Date'] = pd.to_datetime(data['Date'])
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %I-%p', errors='coerce')
    data = data.set_index('Date', drop=False)
    
    data['Open'] = (data['Open'] - data['Open'].min())/(data['Open'].max() - data['Open'].min())
    data['Close'] = (data['Close'] - data['Close'].min())/(data['Close'].max() - data['Close'].min())

    data['rma60_close'] = data['Close'].rolling(60, min_periods=1).mean()
    data['rma60_open'] = data['Open'].rolling(60, min_periods=1).sum()
    data['rma16_close'] = data['Close'].rolling(16, min_periods=1).mean()
    data['rma16_open'] = data['Open'].rolling(16, min_periods=1).sum()

    #plt.figure(figsize=(15,8))
    #sns.lineplot(data=data,x='Date',y='Close',linewidth=0.8)
    #sns.lineplot(data=data,x='Date',y='rma60_close',linewidth=1)
    #sns.lineplot(data=data,x='Date',y='rma16_close',linewidth=1)
    #plt.show(block=False)
    #plt.pause(10)
    #plt.close()

    print(data.index.min(), data.index.max())
    print(data.head())

    date_start = '2019-01-01'
    date_split = '2019-08-01'
    test = data[date_split:]
    train = data[:date_split]
    print(f'train {len(train)} | test {len(test)}')
    #self.numpy_data=scipy.signal.detrend(self.numpy_data)

    # just a short test
    env = QEnvironment(train, hist_t=20)

    Q, total_losses, total_rewards = train_ddqn(env)
    #Q, total_losses, total_rewards = train_dqn(QEnvironment(train))
#    print(f'total loss: {total_losses}, total_rewards: {total_rewards}')
