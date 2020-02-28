
import locale
import pandas as pd
def load_data(filepath):
    data = pd.read_csv(filepath, skiprows=0)
    #data = pd.read_csv('tsla.us.txt')
    #data['Date'] = pd.to_datetime(data['Date'])
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    #data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %I-%p', errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
    data = data.set_index('Date', drop=False)
    
    data['Open'] = (data['Open'] - data['Open'].min())/(data['Open'].max() - data['Open'].min())
    data['Close'] = (data['Close'] - data['Close'].min())/(data['Close'].max() - data['Close'].min())

    #data['rma60_close'] = data['Close'].rolling(60, min_periods=1).mean()
    #data['rma60_open'] = data['Open'].rolling(60, min_periods=1).sum()
    #data['rma16_close'] = data['Close'].rolling(16, min_periods=1).mean()
    #data['rma16_open'] = data['Open'].rolling(16, min_periods=1).sum()

    print(data.index.min(), data.index.max())
    print(data.head())

    date_split = '2019-08-01'
    test = data[date_split:]
    train = data['2013-03-01':date_split]

    return test,train