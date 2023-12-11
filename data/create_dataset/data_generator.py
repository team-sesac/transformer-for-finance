import pandas as pd
import data.create_dataset.utils as u
from src.transformer.model.model_utils import create_directory_if_not_exists
import os



# TA 지표 포함 전체 columns 이름 목록
# ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'pct_change',
#        'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
#        'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
#        'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
#        'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
#        'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
#        'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
#        'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
#        'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
#        'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
#        'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
#        'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
#        'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
#        'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
#        'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
#        'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
#        'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
#        'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
#        'trend_psar_down', 'trend_psar_up_indicator',
#        'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
#        'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
#        'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
#        'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
#        'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
#        'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
#        'others_cr']


def get_stock_codes():
    return u.get_KRX_list().reset_index(drop=True)


def get_dataset_for_transformer(idx, ticker, start_date='20100101', end_date='20231208', out_directory='./'):
    '''
    1개 종목의 2010년~2023년 동안의 주가 + 일부 ta지표 데이터셋을 csv로 저장하는 코드
    ticker.Code = '005380'
    ticker.Name = '삼성전자'
    005380 현대자동차, 000270 기아
    '''

    # 선택할 지표가 있는경우 to_include 주석 해제 (선택 안하면 모든 ta 지표가 데이터셋에 저장됨)
    # to_include = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'pct_change',
    #               'trend_macd_diff', 'trend_sma_fast', 'trend_ema_fast', 'volume_obv',
    #               'momentum_rsi', 'trend_cci', 'volatility_bbh']

    # 1개 종목의 데이터 가져오기
    df = u.get_single_ticker(symbol=ticker.Code, start_date=start_date,
                             end_date=end_date).reset_index()
    try:
        df = u.cat_ta(df)
        # 선택지표만 추출하려면 아래것 주석해제
        # df = u.cat_ta(df, to_include)
    except ValueError:
        return

    # 날짜 빼고 모든 컬럼 datatype float로 바꾸기
    columns_to_convert = df.columns.to_list()
    columns_to_convert.remove('Date')
    df[columns_to_convert] = df[columns_to_convert].astype(float)

    # 종목 이름 첫번재 줄에 쓰기
    df.insert(loc=1, column='Ticker', value=ticker.Name)
    # 상위폴더의 tf_dataset에 01_삼성전자_2010.csv 파일명으로 저장하기
    # df.to_csv(filepath=f'../themed/{idx}_{ticker.Name}_2010.csv', encoding='UTF-8', index=False)
    out_directory = os.getcwd() + f'/{idx}_{ticker.Name}_{since}.csv'

    df.to_csv(filepath=out_directory, encoding='UTF-8', index=False)
    print(f'saved {idx}_{ticker.Name}_{since}.csv')

def save_all_stock_data():
    '''한국 상장 종목별로 10개년 일일 데이터(ta 전체포함)셋 csv 만들기'''
    stocks = get_stock_codes()
    # stocks = pd.read_csv('../stock_list.csv')
    # stocks = stocks.iloc[1207:, :]
    for idx, item in enumerate(stocks.itertuples()):
        # idx += 1207
        try:
            get_dataset_for_transformer(idx, item)
        except:
            continue
        if idx % 10 == 0:
            print(f"saved ~ {idx} {item.Name}")


def save_themed_stock_since_listing_date(stocks, listing, out_directory):
    for idx, ticker in enumerate(stocks.itertuples()):
        get_dataset_for_transformer(idx, ticker, start_date='1990', end_date='20231208', out_directory=out_directory)

        if idx % 10 == 0:
            print(f"saved ~ {idx} {ticker.Name}")


# if __name__ == '__main__':
#     save_all_stock_data()
