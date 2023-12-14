import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch1_clustering.utils import *
import pandas as pd
import numpy as np
from datetime import datetime
# today 'yyyymmdd' 형식으로 포맷팅
today = datetime.now().strftime('%Y%m%d')

base_folder = os.path.join(os.getcwd(), 'src/ch5_cal/')

active_etf_stock_price = 'active_etf_stock_price.csv'
active_etf_stock_price = os.path.join(base_folder, active_etf_stock_price)

###
def save_to_csv(df, filename):
    file_path = os.path.join(base_folder, filename)
    df.to_csv(file_path)
###
def load_csv(filename):
    file_path = os.path.join(base_folder, filename)
    return pd.read_csv(file_path)
###
def load_active_etf_info(file="src/ch5_cal/active_etf_list.csv"):
    return pd.read_csv(file, dtype=str)
###
def load_etf_stock_price_df(file=active_etf_stock_price):
    return pd.read_csv(file, parse_dates=['Date'], index_col='Date')

def save_active_etf_stock_price_df(file=active_etf_stock_price, start_date='20111123', end_date=today):
    stock_list = load_active_etf_info()
    stock = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    stock.to_csv(file)


def calculate_active_etf_returns(target_day=252):
    # Load data
    etf_prices = load_etf_stock_price_df()[-target_day:]
    # etf_pred_prices = load_csv('scaled_close_preds.csv')[-target_day:]
    etf_pred_prices = load_csv('scaled_close_targets.csv')[-target_day:]
    etf_pred_prices.columns = etf_prices.columns
    etf_pred_prices.index = etf_prices.index

    # 매매 신호 계산
    buy_signals = etf_pred_prices.shift(-1) > etf_pred_prices
    buy_signals = buy_signals[:-1]
    # save_to_csv(buy_signals, 'active_buy_signals.csv')
    
    # 일별 수익률
    daily_returns = (etf_prices.shift(-1) - etf_prices)/etf_prices
    daily_returns = daily_returns[:-1]
    # save_to_csv(daily_returns, 'active_etf_daily_returns.csv')

    # 매매 신호에 따른 수익률 필터링
    filtered_returns = daily_returns[buy_signals]
    # save_to_csv(filtered_returns, 'active_etf_filtered_returns.csv')
    
    # 기하평균 누적 수익률 계산
    adjusted_returns = filtered_returns.fillna(0) + 1
    geometric_returns = adjusted_returns.cumprod() - 1
    portfolio_return = geometric_returns.prod(axis=1) ** (1 / len(geometric_returns.columns)) - 1

    save_to_csv(geometric_returns, 'active_etf_geometric_returns.csv')
    save_to_csv(portfolio_return, 'active_etf_portfolio_return.csv')
    return geometric_returns, portfolio_return
    
if __name__ == '__main__':
    # Active
    # save_active_etf_stock_price_df(start_date='20111123', end_date='20231208') # active etf price 
    geometric_returns, portfolio_return = calculate_active_etf_returns()
    print(geometric_returns, portfolio_return)
