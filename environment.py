import math
import numpy as np
from scipy.signal import detrend

class QEnvironment:
    def __init__(self, data, hist_t=1):
        self.fields = ['Open','Close','Volume']
        self.data = data[self.fields].values
        #self.data = detrend(self.data)
        self.log = None
        self.print_log = None
        self.hist_t = hist_t
        
        self.reset()
    
    def len(self):
        return len(self.data) - (self.hist_t+1)
        
    def reset(self):
        self.t = self.hist_t
        self.done = False
        self.profits = 0
        self.positions = []
        self.short_positions = []
        self.position_value = 0
        self.position_value_abs = 0
        self.balance = 100000.0
        self.hold_cnt = 0
        self.max_profits = 0.0
        state_data = np.zeros((self.data.shape[0], 4))
        self.data_combined = np.append(self.data, state_data, axis=1)

        return np.zeros((self.hist_t, self.data_combined.shape[1])) # obs
    
    def step(self, act):
        reward = 0
        
        HOLD = 0
        LONG = 1
        SELL = 2
        SHORT = 3
        CLOSE = 4

        price = self.data[self.t, 1]
        # act = 0: hold, 1: long, 2: sell, 3: short, 4: close
        if act == HOLD:
            self.data_combined[self.t, 3:] = np.array([HOLD, self.balance, len(self.positions), len(self.short_positions)])
        elif act == LONG:
            self.positions.append({'t':self.t, 'price':price})
            self.balance -= price
            self.data_combined[self.t, 3:] = np.array([LONG, self.balance, len(self.positions), len(self.short_positions)])
        elif act == SELL: # sell
            profits = 0
            for p in self.positions:
                profits += price - p['price']
                self.balance += price - p['price']

            self.balance += profits
            reward = 1.0 if profits != 0.0 else 0.0
            
            self.positions = []
            self.data_combined[self.t, 3:] = np.array([SELL, self.balance, len(self.positions), len(self.short_positions)])
        elif act == SHORT:
            self.short_positions.append({'t':self.t, 'price':price})
            self.data_combined[self.t, 3:] = np.array([SHORT, self.balance, len(self.positions), len(self.short_positions)])
        elif act == CLOSE:
            agg_short = 0
            for p in self.short_positions:
                agg_short += (p['price'] - price)
            self.balance += agg_short
            reward = 1.0 if agg_short > 0.0 else 0.0

            self.short_positions = []
            self.data_combined[self.t, 3:] = np.array([CLOSE, self.balance, len(self.positions), len(self.short_positions)])
            
        #debug
        #print(f'act: {act}, positions: {len(self.positions)}')
        # set next time
        self.t += 1
        self.done = self.t == self.len()-1
        return np.array(self.data_combined[self.t-self.hist_t:self.t, :], dtype=np.float32), reward, self.done # obs, reward, done

    def current_obs(self):
        return self.data_combined[self.t-1:self.t, :]
    
    def future_entry(self):
        if self.t+1 < self.len():
            return self.data_combined[self.t+1, 1]
        else:
            return 0.0