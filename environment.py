import math

class QEnvironment:
    
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.best_profit = 1.0
        self.log = None
        self.transaction_idx = 0
        
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.position_value_abs = 0
        self.history = [0 for _ in range(self.history_t)]
        self.balance = 1000.0
        self.transaction_idx = self.transaction_idx + 1

        return [self.position_value, self.balance] + self.history # obs
    
    def step(self, act):
        reward = 0
        if self.log is None:
            self.log = open("transactions.csv", "w+")
        
        HOLD = 0
        BUY = 1
        SELL = 2

        # act = 0: hold, 1: buy, 2: sell
        if act == HOLD:
            if len(self.positions) == 0:
                reward = -1
            self.log.write(f"{self.transaction_idx},HOLD,{self.data.iloc[self.t, :]['Date']},{self.data.iloc[self.t, :]['Close']},{reward},{self.balance}\n")
        elif act == BUY:
            price = self.data.iloc[self.t, :]['Close']
            if self.balance - price < 0:
                reward = -1
                self.done = True
            else:
                self.positions.append(price)
                self.balance -= price

            self.log.write(f"{self.transaction_idx},BUY,{self.data.iloc[self.t, :]['Date']},{self.data.iloc[self.t, :]['Close']},{reward},{self.balance}\n")
        elif act == SELL: # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                    self.balance += self.data.iloc[self.t, :]['Close']
                    reward += float((self.data.iloc[self.t, :]['Close'] - p)>0)
                #print(self.balance, profits)
                self.transaction_idx = self.transaction_idx + 1

                self.profits += profits
                
                # reward clipping -1<r<1
                if profits > 0:
                    # slows down learning a lot, maybe try later with longer training
                    #reward /= len(self.positions)
                    reward = 1
                else:
                    reward = -1
                self.positions = []
                self.log.write(f"{self.transaction_idx},SELL,{self.data.iloc[self.t, :]['Date']},{self.data.iloc[self.t, :]['Close']},{reward},{self.balance}\n")
        
        #debug
        #print(f'act: {act}, positions: {len(self.positions)}')
        # set next time
        self.t += 1
        self.position_value = 0
        self.position_value_abs = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
            self.position_value_abs += self.data.iloc[self.t, :]['Close']
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])
        
        return [self.position_value, self.balance] + self.history, reward, (self.balance <= 0) # obs, reward, done
