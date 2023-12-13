import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch5_cal.utils import *

stock = load_etf_stock_price_df()
print(stock.info())
# import numpy as np

# # Excluding the 'Date' column and selecting only numeric columns
# numeric_columns = df.select_dtypes(include=[np.number]).columns

# # Generating 5% noise for the numeric columns
noise = stock * np.random.normal(0, 0.05, stock.shape)

# # Creating the test dataset with noise added
# test_df = stock.copy()
test_df = stock + noise
test_df.to_csv('etf_stock_pred_price_df.csv')
print(test_df.head())
