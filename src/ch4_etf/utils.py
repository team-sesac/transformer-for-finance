import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch1_clustering.utils import *
import pandas as pd

def extract_portfolio_list_from_cluster():
    df0 = pd.read_csv("src/ch3_ta/cluster0.csv", dtype=str)[['Code', 'Name']]
    df1 = pd.read_csv("src/ch1_clustering/themed_stocks_with_cluster.csv", dtype=str)
    df1 = df1[df1['Cluster'] == '4'][['Code', 'Name']].reset_index(drop=True)
    df = pd.concat([df0, df1], axis=0)
    df.to_csv('src/ch4_etf/portfolio_list_by_cluster.csv', index=False)

def load_portfolio_name(file='src/ch4_etf/portfolio_list_by_cluster.csv'):
    portfolio_list = pd.read_csv(file)['Name'].to_numpy()
    return portfolio_list

def save_close_data_of_portfolio_list(start_date='20111123', end_date='20231212'):
    portfolio_list = load_portfolio_name()
    start_date, end_date = start_date, end_date

    KRX_list = pd.DataFrame(get_KRX_list())
    stock_list = KRX_list[KRX_list['Name'].isin(portfolio_list)]

    price = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    price.dropna(axis=1, inplace=True)
    price.to_csv('src/ch4_etf/portfolio_close_data.csv')


def load_close_data_of_portfolio_list(file='src/ch4_etf/portfolio_close_data.csv'):
    price = pd.read_csv(file, index_col=0)
    return price



if __name__ == '__main__':
    # src/ch1_clustering/themed_stocks_with_cluster.csv
    # src/ch3_ta/cluster0.csv
    # 위 2개 파일을 합치고 close data 저장하고 load 하여 확인
    save_close_data_of_portfolio_list()
    price = load_close_data_of_portfolio_list()
    print(price)