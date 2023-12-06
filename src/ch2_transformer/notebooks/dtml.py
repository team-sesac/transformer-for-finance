#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import os
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List
import lightning as pl

src_path = Path('.').absolute().parent
data_path = src_path / 'data'


# In[337]:


data_path = src_path / 'data'
ps = list((data_path / 'kdd17/price_long_50').glob('*'))
p = ps[0]


# In[301]:

df_date = pd.read_csv('../data/kdd17/trading_dates.csv', header=None)
# df_date = pd.read_csv('../data/trading_dates.csv', header=None)
df = pd.read_csv(p)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
df.drop(columns=df.columns[7], inplace=True)


# In[ ]:


# df_date = pd.read_csv(data_path / 'trading_dates.csv', header=None)


# In[406]:


class DTMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, ds_config):
        self.data_dir = data_path / ds_config['path']

class DTMLDataset(pl.LightningDataModule):
    def __init__(self, data_path, window_size, dtype=''):

        # ref: https://arxiv.org/abs/1810.09936
        ds_info = {
            # train: (Jan-01-2007 to Jan-01-2015)
            # val: (Jan-01-2015 to Jan-01-2016)
            # test: (Jan-01-2016 to Jan-01-2017)
            'kdd17': {
                'path': 'kdd17/price_long_50',
                'date_path': 'kdd17/trading_dates.csv',
                'train_date': '2015-01-01', 
                'val_date': '2016-01-01', 
                'test_date': '2017-01-01',
                'window_size': window_size
            },
            # train: (Jan-01-2014 to Aug-01-2015)
            # vali: (Aug-01-2015 to Oct-01-2015)
            # test: (Oct-01-2015 to Jan-01-2016)
            'acl18': {
                'path': 'stocknet-dataset/price/raw',
                'date_path': 'stocknet-dataset/price/trading_dates.csv',
                'train_date': '2015-08-01', 
                'val_date': '2015-10-01', 
                'test_date': '2016-01-01',
                'window_size': window_size
            }
        }
        ds_config = ds_info[dtype]
    
    def load_dataset(self, config: dict):
        tick_files = [p for p in self.data_dir.glob('*') if not p.is_dir()]

        train_data = []
        val_data = []
        test_data = []
        for p in tick_files:
            df = self.load_single_tick(p)
            # train / val / test split
            df['date']

    def load_single_tick(self, p: Path | str):
        def longterm_trend(x: pd.Series, k:int):
            return (x.rolling(k).sum().div(k*x) - 1) * 100

        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if 'Unnamed' in df.columns:
            df.drop(columns=df.columns[7], inplace=True)
        if 'Original_Open' in df.columns:
            df.rename(columns={'Original_Open': 'Open', 'Open': 'Adj Open'}, inplace=True)

        # Open, High, Low
        z1 = (df.loc[:, ['Open', 'High', 'Low']].div(df['Close'], axis=0) - 1).rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low'}) * 100
        # Close
        z2 = df[['Close']].pct_change().rename(columns={'Close': 'close'}) * 100
        # Adj Close
        z3 = df[['Adj Close']].pct_change().rename(columns={'Adj Close': 'adj_close'}) * 100

        z4 = []
        for k in [5, 10, 15, 20, 25, 30]:
            z4.append(df[['Adj Close']].apply(longterm_trend, k=k).rename(columns={'Adj Close': f'zd{k}'}))

        df_pct = pd.concat([df['Date'], z1, z2, z3] + z4, axis=1).rename(columns={'Date': 'date'})
        cols_max = df_pct.columns[df_pct.isnull().sum() == df_pct.isnull().sum().max()]
        df_pct = df_pct.loc[~df_pct[cols_max].isnull().values, :]

        # from https://arxiv.org/abs/1810.09936
        # Examples with movement percent ≥ 0.55% and ≤ −0.5% are 
        # identified as positive and negative examples, respectively
        
        df_pct['label'] = 0
        df_pct.loc[(df_pct['close'] >= 0.55), 'label'] = 1
        df_pct.loc[(df_pct['close'] <= -0.5), 'label'] = -1

        # only select rise / fall
        df_pct['label'] = (df_pct['label'] + 1) / 2
        df_pct = df_pct.loc[df_pct['label'] != 0.5, :].reset_index(drop=True)

        return df_pct


# In[415]:


def load_single_tick(p: Path | str):
    def longterm_trend(x: pd.Series, k:int):
        return (x.rolling(k).sum().div(k*x) - 1) * 100

    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'])#, format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Unnamed' in df.columns:
        df.drop(columns=df.columns[7], inplace=True)
    if 'Original_Open' in df.columns:
        df.rename(columns={'Open': 'Adj Open', 'Original_Open': 'Open'}, inplace=True)
    # Open, High, Low
    z1 = (df.loc[:, ['Open', 'High', 'Low']].div(df['Close'], axis=0) - 1).rename(
        columns={'Open': 'open', 'High': 'high', 'Low': 'low'}) * 100
    # Close
    z2 = df[['Close']].pct_change().rename(columns={'Close': 'close'}) * 100
    # Adj Close
    z3 = df[['Adj Close']].pct_change().rename(columns={'Adj Close': 'adj_close'}) * 100

    z4 = []
    for k in [5, 10, 15, 20, 25, 30]:
        z4.append(df[['Adj Close']].apply(longterm_trend, k=k).rename(columns={'Adj Close': f'zd{k}'}))

    df_pct = pd.concat([df['Date'], z1, z2, z3] + z4, axis=1).rename(columns={'Date': 'date'})
    cols_max = df_pct.columns[df_pct.isnull().sum() == df_pct.isnull().sum().max()]
    df_pct = df_pct.loc[~df_pct[cols_max].isnull().values, :]

    # from https://arxiv.org/abs/1810.09936
    # Examples with movement percent ≥ 0.55% and ≤ −0.5% are 
    # identified as positive and negative examples, respectively
    
    df_pct['label'] = 0
    df_pct.loc[(df_pct['close'] >= 0.55), 'label'] = 1
    df_pct.loc[(df_pct['close'] <= -0.5), 'label'] = -1

    # only select rise / fall
    df_pct['label'] = (df_pct['label'] + 1) / 2
    df_pct = df_pct.loc[df_pct['label'] != 0.5, :].reset_index(drop=True)

    return df_pct


# In[416]:


window_size = 5
dtype = 'kdd17'
ds_info = {
    # train: (Jan-01-2007 to Jan-01-2015)
    # val: (Jan-01-2015 to Jan-01-2016)
    # test: (Jan-01-2016 to Jan-01-2017)
    'kdd17': {
        'path': 'kdd17/price_long_50',
        'date_path': 'kdd17/trading_dates.csv',
        'train_date': '2015-01-01', 
        'val_date': '2016-01-01', 
        'test_date': '2017-01-01',
        'window_size': window_size
    },
    # train: (Jan-01-2014 to Aug-01-2015)
    # vali: (Aug-01-2015 to Oct-01-2015)
    # test: (Oct-01-2015 to Jan-01-2016)
    'acl18': {
        'path': 'stocknet-dataset/price/raw',
        'date_path': 'stocknet-dataset/price/trading_dates.csv',
        'train_date': '2015-08-01', 
        'val_date': '2015-10-01', 
        'test_date': '2016-01-01',
        'window_size': window_size
    }
}
ds_config = ds_info[dtype]


# In[379]:


index2date = pd.read_csv(data_path / ds_config['date_path'], header=None).to_dict()[0]


# In[417]:


data_dir = data_path / ds_config['path']
tick_files = [p for p in data_dir.glob('*') if not p.is_dir()]


# In[418]:


data = []
for p in tick_files:
    data.append(load_single_tick(p))


# In[380]:


df = load_single_tick(p)
train_idx = df['date'] < ds_config['train_date']
val_idx = (ds_config['train_date'] <= df['date']) & (df['date'] < ds_config['val_date'])
test_idx = (ds_config['val_date'] <= df['date']) & (df['date'] < ds_config['test_date'])

print(train_idx.sum(), val_idx.sum(), test_idx.sum())

train_data = df.loc[train_idx, :]
val_data = df.loc[val_idx, :]
test_idx = df.loc[test_idx, :]


# In[349]:


def sliding_window(T: int, window_size: int, time_lag: int):
    X_idx = np.expand_dims(np.arange(window_size), 0) + \
                np.expand_dims(np.arange(T - window_size - time_lag + 1), 0).T
    y_idx = np.arange(time_lag + window_size - 1 , T)
    
    return X_idx, y_idx


# In[350]:


T = len(df.loc[train_idx, :])
window_size = 10
time_lag = 1


# In[366]:
X_idx, y_idx = sliding_window(T=T, window_size=window_size, time_lag=time_lag)

print(X_idx[0], y_idx[0])


# In[351]:


X_idx, y_idx = sliding_window(T, window_size, time_lag)


# In[361]:


y_idx


# In[360]:


train_data.values[X_idx].shape


# In[172]:


import torch
import torch.nn as nn
import pytorch_lightning as pl

class TimeAxisAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn=False):
        # x: (D, W, L)
        o, (h, _) = self.lstm(x) # o: (D, W, H) / h: (1, D, H)
        score = torch.bmm(o, h.permute(1, 2, 0)) # (D, W, H) x (D, H, 1)
        tx_attn = torch.softmax(score, 1).squeeze(-1)  # (D, W)
        context = torch.bmm(tx_attn.unsqueeze(1), o).squeeze(1)  # (D, 1, W) x (D, W, H)
        normed_context = self.lnorm(context)
        if rt_attn:
            return normed_context, tx_attn
        else:
            return normed_context, None
            
class DataAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_rate=0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, hm: torch.tensor, rt_attn=False):
        # Forward Multi-head Attention
        residual = hm
        # hm_hat: (D, H), dx_attn: (D, D) 
        hm_hat, dx_attn = self.multi_attn(hm, hm, hm)
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))

        # Forward FFN
        residual = hm_hat
        # hp: (D, H)
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))

        if rt_attn:
            return hp, dx_attn
        else:
            return hp, None

class DTML(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_heads, beta=0.1, drop_rate=0.1):
        super.__init__()
        self.beta = beta
        self.txattention = TimeAxisAttention(input_size, hidden_size, num_layers)
        self.dxattention = DataAxisAttention(hidden_size, n_heads, drop_rate)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, stocks, index, rt_attn=False):
        # stocks: (W, D, L) for a single time stamp
        # index: (W, 1, L) for a single time stamp
        # W: length of observations
        # D: number of stocks
        # L: number of features
        
        # Time-Axis Attention
        # c_stocks: (D, H) / tx_attn_stocks: (D, W)
        c_stocks, tx_attn_stocks = self.txattention(stocks.transpose(1, 0), rt_attn=rt_attn)
        # c_index: (1, H) / tx_attn_index: (1, W)
        c_index, tx_attn_index = self.txattention(index.transpose(1, 0), rt_attn=rt_attn)
        
        # Context Aggregation
        # Multi-level Context
        # hm: (D, H)
        hm = c_stocks + self.beta * c_index
        # The Effect of Global Contexts
        # effect: (D, D)
        effect = c_stocks.mm(c_stocks.transpose(0, 1)) + \
            self.beta * c_index.mm(c_stocks.transpose(1, 0)) + \
            self.beta**2 * torch.mm(c_index, c_index.transpose(0, 1)) 

        # Data-Axis Attention
        # hp: (D, H) / dx_attn: (D, D)
        hp, dx_attn_stocks = self.dxattention(hm, rt_attn=rt_attn)
        # output: (D, 1)
        output = self.linear(hp)

        return {
            'output': output,
            'tx_attn_stocks': tx_attn_stocks,
            'tx_attn_index': tx_attn_index,
            'dx_attn_stocks': dx_attn_stocks,
            'effect': effect
        }

