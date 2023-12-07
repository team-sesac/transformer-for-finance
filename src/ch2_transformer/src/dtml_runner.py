import torch
import os
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List
import lightning as pl




if __name__ == '__main__':
    # 데이터 로드
    data_path = Path('.').absolute().parent / 'data'
    p = data_path / "kdd17/price_long_50/AAPL.csv"
    df_date = pd.read_csv('../data/kdd17/trading_dates.csv', header=None)
    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    df.drop(columns=df.columns[7], inplace=True)

