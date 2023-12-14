import pandas as pd

if __name__ == '__main__':
    df_ori = pd.read_csv('active_etf_geometric_returns.csv', encoding='utf-8')
    df_ori = df_ori.replace(to_replace='\u110b', value='')
    df = df_ori.copy()
    df_list = list()
    for i in range(df.shape[1]-1):
        curr_idx = i+1
        curr_name = df.columns[curr_idx]
        curr_df = df[['Date', curr_name]]
        curr_df['Ticker'] = f"{curr_name}"
        curr_df.columns = ['Date', 'ReturnRate', 'Ticker']
        curr_df = curr_df[['Date', 'Ticker', 'ReturnRate']]
        df_list.append(curr_df)

    final_df = pd.concat(df_list)
    final_df.to_csv('active_etf_geometric_returns_concatenated.csv', index=False, encoding='UTF-8')
    print('here')