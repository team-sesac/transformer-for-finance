from utils import *

start_date, end_date = '20160101', '20231130'
all_stocks = get_dataset(start_date, end_date)
all_stocks.to_csv("../../data/krx_2016.csv")