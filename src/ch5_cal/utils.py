import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch1_clustering.utils import *
import pandas as pd
from datetime import datetime
# today 'yyyymmdd' 형식으로 포맷팅
today = datetime.now().strftime('%Y%m%d')
etf_stock_price_df = 'src/ch5_cal/etf_stock_price_df.csv'

def load_etf_info(file="portfolio/optimized_portfolio_ratio.csv"):
    etf_list = pd.read_csv(file, index_col=0)
    return etf_list

def save_etf_stock_price_df(file=etf_stock_price_df, start_date='20231201', end_date=today):
    etf_list = load_etf_info().index
    KRX_list = pd.DataFrame(get_KRX_list())
    stock_list = KRX_list[KRX_list['Name'].isin(etf_list)]
    stock = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    stock.to_csv(file)

def load_etf_stock_price_df(file=etf_stock_price_df):
    stock = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
    return stock

def calculate_etf_returns():
    """
    Calculate and return the ETF stock prices, individual stock returns, 
    total return without portfolio ratio, and total return with portfolio ratio.
    """
    # Load data
    optimized_portfolio_ratio_df = load_etf_info()
    etf_stock_price_df = load_etf_stock_price_df().iloc[[0, -1]]

    # Calculate returns for each stock
    returns = ((etf_stock_price_df.iloc[1, 1:] - etf_stock_price_df.iloc[0, 1:]) 
               / etf_stock_price_df.iloc[0, 1:]) * 100

    # Total return without portfolio ratio
    total_return_without_ratio = returns.values.mean()

    # Align and calculate weighted returns for total return with portfolio ratio
    aligned_weights = optimized_portfolio_ratio_df.loc[returns.index].values
    weighted_returns = returns.values * aligned_weights
    total_return_with_ratio = weighted_returns.sum() / len(weighted_returns)

    return etf_stock_price_df, returns, total_return_without_ratio, total_return_with_ratio


def cal_etf_total_return():
    pass    

if __name__ == '__main__':
    # save_etf_stock_price_df(start_date='20231201')
    # save_etf_stock_price_df(start_date='20230101', end_date='20231212')
    # load_etf_stock_price_df()
    etf_stock_price_df, returns, total_return_without_ratio, total_return_with_ratio = calculate_etf_returns()
    print(etf_stock_price_df, returns, total_return_without_ratio, total_return_with_ratio)