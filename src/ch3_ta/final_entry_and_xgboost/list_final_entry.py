import pandas as pd
# 고위험 투자 테마주 목록

if __name__ == '__main__':
    # df = pd.read_csv('./all_final_entry_stock.csv', dtype=str)
    # df_no_duplicates = df.drop_duplicates(subset='Code')
    # df = df_no_duplicates[['Ticker', 'Code']]
    # df.to_csv('list_final_entry.csv', index=False, encoding='UTF-8')
    df = pd.read_csv('./list_final_entry.csv', dtype=str)
    listing_df = pd.read_csv('../../ch3_ta/stock_listing_date.csv', dtype=str)
    merged = pd.merge(left=df, right=listing_df, left_on='Code', right_on='종목코드')

    theme_df = pd.read_csv('../../ch3_ta/themed_stocks_with_code_no_dup.csv', dtype=str)
    merged2 = pd.merge(left=merged, right=theme_df, on='Code')

    merged2.to_csv('final_entry_basic_info.csv', index=False, encoding='UTF-8')
    print('here')