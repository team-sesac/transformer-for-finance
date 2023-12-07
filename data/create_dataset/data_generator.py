import pandas as pd

import utils as u

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

def get_dataset_for_transformer():
    # data_list = u.get_KRX_list()
    # 005380 현대자동차
    # 000270 기아
    to_include = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'pct_change',
                  'trend_macd_diff', 'trend_sma_fast', 'trend_ema_fast', 'volume_obv',
                  'momentum_rsi', 'trend_cci', 'volatility_bbh']

    df1 = u.get_single_ticker(symbol='005380',
                              start_date='20100101',
                              end_date='20231130')
    df1 = u.cat_ta(df1, to_include)

    df2 = u.get_single_ticker(symbol='000270',
                              start_date='20100101',
                              end_date='20231130')
    df2 = u.cat_ta(df2, to_include)
    fin_df = pd.concat([df1, df2], axis=1)

    u.export_csv(fin_df, '../hyundai_kia_2010.csv')


if __name__ == '__main__':
    get_dataset_for_transformer()

