import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch1_clustering.utils import *
import pandas as pd
import numpy as np
from datetime import datetime
# today 'yyyymmdd' 형식으로 포맷팅
today = datetime.now().strftime('%Y%m%d')

base_folder = os.path.join(os.getcwd(), 'src/ch5_cal/')

passive_etf_stock_price_df = 'passive_etf_stock_price_df.csv'
passive_etf_stock_price_df = os.path.join(base_folder, passive_etf_stock_price_df)


def save_to_csv(df, filename):
    file_path = os.path.join(base_folder, filename)
    df.to_csv(file_path)

def load_csv(filename):
    file_path = os.path.join(base_folder, filename)
    df = pd.read_csv(file_path)
    return df

def load_passive_etf_info(file="portfolio/optimized_portfolio_ratio.csv"):
    etf_list = pd.read_csv(file, index_col=0)
    return etf_list

def save_passive_etf_stock_price_df(file=passive_etf_stock_price_df, start_date='20220101', end_date=today):
    etf_list = load_passive_etf_info().index
    KRX_list = pd.DataFrame(get_KRX_list())
    stock_list = KRX_list[KRX_list['Name'].isin(etf_list)]
    stock = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    stock.to_csv(file)

def load_etf_stock_price_df(file=passive_etf_stock_price_df):
    return pd.read_csv(file, parse_dates=['Date'], index_col='Date')

def calculate_passive_etf_returns(target_day=2):
    """
    Calculate and return the ETF stock prices, individual stock returns, 
    total return without portfolio ratio, and total return with portfolio ratio.

    target_day=2
    비교 대상 일자.
    2 이상의 숫자.
    """
    # Load data
    etf_price = load_etf_stock_price_df().iloc[[-target_day, -1]]
    ratio = load_passive_etf_info().iloc[:,0]

    # Calculate returns for each stock
    daily_returns = ((etf_price.iloc[1,:] - etf_price.iloc[0,:]) 
               / etf_price.iloc[0,:]) * 100

    # Total return without portfolio ratio
    total_return_without_ratio = daily_returns.values.mean()

    # calculate weighted returns for total return with portfolio ratio
    weighted_returns = daily_returns.values * ratio
    total_return_with_ratio = weighted_returns.sum()

    return total_return_without_ratio, total_return_with_ratio

def calculate_passive_etf_returns_days():
    m = 21
    mon = [2, 5, m, m*3, m*6, m*12] # 1일 1주일 1개월 3개월 6개월 12개월
    total_return_without_ratio, total_return_with_ratio = [], []

    for _m in mon:
        return_without_ratio, return_with_ratio = calculate_passive_etf_returns(target_day=_m)
        total_return_without_ratio.append(return_without_ratio)
        total_return_with_ratio.append(return_with_ratio)

    passive_etf_returns = pd.DataFrame(data=[total_return_without_ratio, total_return_with_ratio],
                      index=['return_without_ratio', 'return_with_ratio'],
                      columns=['1-day', '1-week', '1-month', '3-month', '6-month', '12-month'])
    
    save_to_csv(passive_etf_returns, 'passive_etf_returns.csv')
    return passive_etf_returns


if __name__ == '__main__':
    # Passive
    save_passive_etf_stock_price_df()
    df = calculate_passive_etf_returns_days()
    print(df)
