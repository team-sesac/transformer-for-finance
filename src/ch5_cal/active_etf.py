import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch1_clustering.utils import *
import pandas as pd
import numpy as np
from datetime import datetime
# today 'yyyymmdd' 형식으로 포맷팅
today = datetime.now().strftime('%Y%m%d')

base_folder = os.path.join(os.getcwd(), 'src/ch5_cal/')

active_etf_stock_price_df = 'active_etf_stock_price_df.csv'
active_etf_stock_price_df = os.path.join(base_folder, active_etf_stock_price_df)

def _save_file(df, filename):
    file_path = os.path.join(base_folder, filename)
    df.to_csv(file_path)

def _load_csv(filename):
    file_path = os.path.join(base_folder, filename)
    df = pd.read_csv(file_path)
    return df

def load_active_etf_info(file="src/ch5_cal/active_etf_list.csv"):
    etf_list = pd.read_csv(file, dtype=str)
    return etf_list

def load_etf_stock_price_df(file=active_etf_stock_price_df):
    stock = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
    return stock

def save_active_etf_stock_price_df(file=active_etf_stock_price_df, start_date='20111124', end_date=today):
    stock_list = load_active_etf_info()
    stock = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    stock.to_csv(file)


def calculate_active_etf_returns(target_day=252):
    # Load data
    active_etf_stock_price_df = load_etf_stock_price_df()[-target_day:]
    active_etf_stock_pred_price_df = _load_csv('scaled_close_targets.csv')[-target_day:]
    active_etf_stock_pred_price_df.columns = active_etf_stock_price_df.columns
    active_etf_stock_pred_price_df.index = active_etf_stock_price_df.index

    # 매매 판단 인덱스. True = 매수, False = 킵
    active_etf_stock_pred_price_df_index = active_etf_stock_pred_price_df.shift(-1) > active_etf_stock_pred_price_df
    # 마지막행은 NaN으로 제외
    active_etf_stock_pred_price_df_index = active_etf_stock_pred_price_df_index[:-1]
    _save_file(active_etf_stock_pred_price_df_index, 'active_etf_stock_pred_price_df_index.csv')
    
    # 일자별 수익률
    active_etf_stock_price_df_increment = (active_etf_stock_price_df.shift(-1) - active_etf_stock_price_df)/active_etf_stock_price_df
    # 마지막행은 NaN으로 제외
    active_etf_stock_price_df_increment = active_etf_stock_price_df_increment[:-1]
    _save_file(active_etf_stock_price_df_increment, 'active_etf_stock_price_df_increment.csv')

    # 일자별 수익률[매매 판단 인덱스]
    active_etf_stock_price_df_increment_cumsum = active_etf_stock_price_df_increment[active_etf_stock_pred_price_df_index]
    _save_file(active_etf_stock_price_df_increment_cumsum, 'active_etf_stock_price_df_increment_cumsum.csv')
    
    # 종목별 누적 수익률 cumsum()
    cumulative_returns = active_etf_stock_price_df_increment_cumsum.fillna(0).cumsum()
    # 전체 누적 수익률 : 종목별 누적 수익률의 평균.(동일한 비율로 투자 한다는 가정)
    portfolio_cumulative_return = cumulative_returns.mean(axis=1)

    _save_file(cumulative_returns, 'active_etf_cumulative_returns.csv')
    _save_file(portfolio_cumulative_return, 'active_etf_portfolio_cumulative_return.csv')
    return cumulative_returns.tail(), portfolio_cumulative_return.tail()



if __name__ == '__main__':
    # Active
    # save_active_etf_stock_price_df(start_date='20111123', end_date='20231208') # active etf price 
    cumulative_returns, portfolio_cumulative_return = calculate_active_etf_returns()
    # cumulative_returns, portfolio_cumulative_return = calculate_active_etf_returns(target_day=21)
    print(cumulative_returns, portfolio_cumulative_return)
