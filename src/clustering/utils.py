import ta
import FinanceDataReader as fdr
import pandas as pd
import os

# 한국거래소 상장종목 전체 조회
def get_KRX_list():
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    kospi_list = kospi[['Code', 'Name']]
    kosdaq_list = kosdaq[['Code', 'Name']]
    data_list = pd.concat([kospi_list, kosdaq_list], axis=0)
    return data_list

# 종목을 start_date ~ end_date 기간 데이터
def get_stock_data(stock_code, stock_name, start_date, end_date):
    stock_df = fdr.DataReader(stock_code, start_date, end_date).reset_index()
    stock_df.insert(0, 'Code', [f'{stock_code}'] * stock_df.shape[0])
    stock_df.insert(0, 'Name', [f'{stock_name}'] * stock_df.shape[0])
    return stock_df

# 코스피, 코스닥 데이터 + 보조지표. 
def get_dataset(start_date, end_date):
    data_list = get_KRX_list()
    all_stocks = pd.DataFrame()
    for code, name in zip(data_list['Code'], data_list['Name']):
        stock = get_stock_data(code, name, start_date, end_date)
        # stock = add_full_ta(stock)
        all_stocks = pd.concat([all_stocks, stock], ignore_index=True)
    # 데이터 계층화를 위한 Date를 index로 작업
    all_stocks.set_index(['Date'], inplace=True)
    all_stocks.index.name = None
    return all_stocks

def get_test_dataset(start_date, end_date):
    data_list = get_KRX_list()[:50]
    all_stocks = pd.DataFrame()
    for code, name in zip(data_list['Code'], data_list['Name']):
        stock = get_stock_data(code, name, start_date, end_date)
        stock = add_full_ta(stock)
        all_stocks = pd.concat([all_stocks, stock], ignore_index=True)
    # 데이터 계층화를 위한 Date를 index로 작업
    all_stocks.set_index(['Date'], inplace=True)
    all_stocks.index.name = None
    return all_stocks

# features = ['지표1', '지표2', '지표3'] # select
# features = all_stocks.columns.drop(['Name']) # drop
# all_stocks[features]


def add_full_ta(stock_df):
    ma = [5,20,60,120]
    for days in ma:
        stock_df['ma_'+str(days)] = stock_df['Close'].rolling(window = days).mean()
    H, L, C, V = stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume']

    # stock_df['bol_high'] = ta.volatility.bollinger_hband(C)
    # stock_df['bol_low']  = ta.volatility.bollinger_lband(C)
    stock_df['MFI'] = ta.volume.money_flow_index(
        high=H, low=L, close=C, volume=V, fillna=True)

    stock_df['ADI'] = ta.volume.acc_dist_index(
        high=H, low=L, close=C, volume=V, fillna=True)

    stock_df['OBV'] = ta.volume.on_balance_volume(close=C, volume=V, fillna=True)
    stock_df['CMF'] = ta.volume.chaikin_money_flow(
        high=H, low=L, close=C, volume=V, fillna=True)

    stock_df['FI'] = ta.volume.force_index(close=C, volume=V, fillna=True)
    stock_df['EOM, EMV'] = ta.volume.ease_of_movement(
        high=H, low=L, volume=V, fillna=True)

    stock_df['VPT'] = ta.volume.volume_price_trend(close=C, volume=V, fillna=True)
    stock_df['NVI'] = ta.volume.negative_volume_index(close=C, volume=V, fillna=True)
    stock_df['VMAP'] = ta.volume.volume_weighted_average_price(
        high=H, low=L, close=C, volume=V, fillna=True)

    # Volatility
    stock_df['ATR'] = ta.volatility.average_true_range(
        high=H, low=L, close=C, fillna=True)
    stock_df['BHB'] = ta.volatility.bollinger_hband(close=C, fillna=True)
    stock_df['BLB'] = ta.volatility.bollinger_lband(close=C, fillna=True)
    stock_df['KCH'] = ta.volatility.keltner_channel_hband(
        high=H, low=L, close=C, fillna=True)
    stock_df['KCL'] = ta.volatility.keltner_channel_lband(
        high=H, low=L, close=C, fillna=True)
    stock_df['KCM'] = ta.volatility.keltner_channel_mband(
        high=H, low=L, close=C, fillna=True)
    stock_df['DCH'] = ta.volatility.donchian_channel_hband(
        high=H, low=L, close=C, fillna=True)
    stock_df['DCL'] = ta.volatility.donchian_channel_lband(
        high=H, low=L, close=C, fillna=True)
    stock_df['DCM'] = ta.volatility.donchian_channel_mband(
        high=H, low=L, close=C, fillna=True)
    stock_df['UI'] = ta.volatility.ulcer_index(close=C, fillna=True)
    # Trend
    stock_df['SMA'] = ta.trend.sma_indicator(close=C, fillna=True)
    stock_df['EMA'] = ta.trend.ema_indicator(close=C, fillna=True)
    stock_df['WMA'] = ta.trend.wma_indicator(close=C, fillna=True)
    stock_df['MACD'] = ta.trend.macd(close=C, fillna=True)
    # stock_df['ADX'] = ta.trend.adx(high=H, low=L, close=C, fillna=True)
    stock_df['-VI'] = ta.trend.vortex_indicator_neg(
        high=H, low=L, close=C, fillna=True)
    stock_df['+VI'] = ta.trend.vortex_indicator_pos(
        high=H, low=L, close=C, fillna=True)
    stock_df['TRIX'] = ta.trend.trix(close=C, fillna=True)
    stock_df['MI'] = ta.trend.mass_index(high=H, low=L, fillna=True)
    stock_df['CCI'] = ta.trend.cci(high=H, low=L, close=C, fillna=True)
    stock_df['DPO'] = ta.trend.dpo(close=C, fillna=True)
    stock_df['KST'] = ta.trend.kst(close=C, fillna=True)
    stock_df['Ichimoku'] = ta.trend.ichimoku_a(high=H, low=L, fillna=True)
    stock_df['Parabolic SAR'] = ta.trend.psar_down(
        high=H, low=L, close=C, fillna=True)
    stock_df['STC'] = ta.trend.stc(close=C, fillna=True)
    # Momentum
    stock_df['RSI'] = ta.momentum.rsi(close=C, fillna=True)
    stock_df['SRSI'] = ta.momentum.stochrsi(close=C, fillna=True)
    stock_df['TSI'] = ta.momentum.tsi(close=C, fillna=True)
    stock_df['UO'] = ta.momentum.ultimate_oscillator(
        high=H, low=L, close=C, fillna=True)
    stock_df['SR'] = ta.momentum.stoch(close=C, high=H, low=L, fillna=True)
    stock_df['WR'] = ta.momentum.williams_r(high=H, low=L, close=C, fillna=True)
    stock_df['AO'] = ta.momentum.awesome_oscillator(high=H, low=L, fillna=True)
    stock_df['KAMA'] = ta.momentum.kama(close=C, fillna=True)
    stock_df['ROC'] = ta.momentum.roc(close=C, fillna=True)
    stock_df['PPO'] = ta.momentum.ppo(close=C, fillna=True)
    stock_df['PVO'] = ta.momentum.pvo(volume=V, fillna=True)
    return stock_df




if __name__ == "__main__":
    file = '../../data/krx_ta_2016.csv'
    file_path = os.path.join(os.path.dirname(__file__), file)
    start_date, end_date = '20160101', '20231206'
    df = get_test_dataset(start_date, end_date)
    print(df)
    # df.to_csv(file_path)