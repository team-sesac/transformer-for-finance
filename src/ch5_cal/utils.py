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

def save_etf_stock_price_df(file=etf_stock_price_df, start_date='20220101', end_date=today):
    etf_list = load_etf_info().index
    KRX_list = pd.DataFrame(get_KRX_list())
    stock_list = KRX_list[KRX_list['Name'].isin(etf_list)]
    stock = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    stock.to_csv(file)

def load_etf_stock_price_df(file=etf_stock_price_df):
    stock = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
    return stock

def calculate_etf_returns(target_day=2):
    """
    Calculate and return the ETF stock prices, individual stock returns, 
    total return without portfolio ratio, and total return with portfolio ratio.

    target_day=2
    비교 대상 일자.
    2 이상의 숫자.
    """
    # Load data
    optimized_portfolio_ratio_df = load_etf_info()
    etf_stock_price_df = load_etf_stock_price_df().iloc[[-target_day, -1]]

    # Calculate returns for each stock
    returns = ((etf_stock_price_df.iloc[1,:] - etf_stock_price_df.iloc[0,:]) 
               / etf_stock_price_df.iloc[0,:]) * 100

    # Total return without portfolio ratio
    total_return_without_ratio = returns.values.mean()

    # calculate weighted returns for total return with portfolio ratio
    weights = optimized_portfolio_ratio_df.iloc[:,0]
    weighted_returns = returns.values * weights
    total_return_with_ratio = weighted_returns.sum()

    return total_return_without_ratio, total_return_with_ratio


def calculate_active_etf_returns(target_day=2, loss=0.003, fee=0.001):
    # Load data
    act_stock_price_df = load_etf_stock_price_df().iloc[[-target_day]] # 전달 받은 파일의 종목 종가
    act_stock_pred_price_df = load_etf_stock_price_df().iloc[[-target_day]] # 전달 받은 파일로 변경
    
    # 모델의 loss 만큼 보수적으로 계산, 매도 수수료 적용
    act_stock_pred_price_df = act_stock_pred_price_df * (1-loss) * (1-fee)

    # Calculate returns for each stock
    returns = ((act_stock_pred_price_df.iloc[0,:] - act_stock_price_df.iloc[0,:]) 
               / act_stock_price_df.iloc[0,:]) * 100

    # 매수 판단
    # 1 = 매수, 0 = 보류
    weights = np.array([act_stock_pred_price_df.iloc[0,:] > act_stock_price_df.iloc[0,:]])
    
    # # 계산
    weighted_returns = returns.values * weights
    total_return_with_ratio = weighted_returns.sum()
    return total_return_with_ratio
    

if __name__ == '__main__':
    # save_etf_stock_price_df()
    # save_etf_stock_price_df(start_date='20230101', end_date='20231212')

    # etf_stock_price_df, returns, total_return_without_ratio, total_return_with_ratio = calculate_etf_returns(4)
    # print(etf_stock_price_df, returns, total_return_without_ratio, total_return_with_ratio)
    # total_return_without_ratio, total_return_with_ratio = calculate_etf_returns(4)
    # print(total_return_without_ratio, total_return_with_ratio)
    total_return_with_ratio = calculate_active_etf_returns()
    print(total_return_with_ratio)