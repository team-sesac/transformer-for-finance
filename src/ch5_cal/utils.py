import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch1_clustering.utils import *
import pandas as pd
from datetime import datetime
# today 'yyyymmdd' 형식으로 포맷팅
today = datetime.now().strftime('%Y%m%d')


def load_etf_info(file="portfolio/optimized_portfolio_ratio.csv"):
    etf_list = pd.read_csv(file)
    return etf_list

def cal_etf_return(start_date='20231201', end_date=today):
    etf_list = load_etf_info()

    KRX_list = pd.DataFrame(get_KRX_list())
    stock_list = KRX_list[KRX_list['Name'].isin(etf_list).iloc[:,0].to_numpy()]

    price = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    price.to_csv('src/ch5_cal/etf_return.csv')



if __name__ == '__main__':
    cal_etf_return()