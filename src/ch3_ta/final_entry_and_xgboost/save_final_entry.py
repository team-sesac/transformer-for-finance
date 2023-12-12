from data.create_dataset.data_generator import save_themed_stock_since_yg_listing_date
import pandas as pd




if __name__ == '__main__':
    stocks = pd.read_csv('../cluster0_final_entry.csv', encoding='UTF-8', dtype=str)
    start_date='20111123' # yg ent. listing date
    save_themed_stock_since_yg_listing_date(stocks, start_date)