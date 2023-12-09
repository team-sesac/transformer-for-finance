import pickle

import pandas as pd
import xgboost as xgb
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train_and_save_xgboost():
    # 피클 파일 로드
    with open('concatenated_array500.pkl', 'rb') as file:
        loaded_object = pickle.load(file)

    X = loaded_object[:, 6:]
    y = loaded_object[:, 5]


    # 데이터를 학습용과 테스트용으로 나눔
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost 회귀 모델 정의 및 학습
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    model.fit(X_train, y_train)

    with open('xgboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)
        print('saved model')

def visualize_feature_importance():
    col_names = ['Volume', 'Change', 'pct_change', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi',
                 'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi',
                 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
                 'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                 'volatility_kcw', 'volatility_kcp', 'volatility_kchi', 'volatility_kcli', 'volatility_dcl',
                 'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp', 'volatility_atr',
                 'volatility_ui', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg',
                 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_sig',
                 'trend_kst_diff', 'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                 'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci', 'trend_visual_ichimoku_a',
                 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi',
                 'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi', 'momentum_uo',
                 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_roc',
                 'momentum_ppo',
                 'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal', 'momentum_pvo_hist',
                 'momentum_kama', 'others_dr', 'others_dlr', 'others_cr']

    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # 특성 중요도 얻기
    feature_importance = model.feature_importances_

    # 각 특성의 중요도를 튜플로 묶어 내림차순으로 정렬
    feature_importance = sorted(zip(col_names, feature_importance), key=lambda x: x[1], reverse=True)

    # 중요도 출력
    df = pd.DataFrame(feature_importance)
    df.to_csv('feature_importance_by_close.csv')

    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")

    # 중요도 시각화
    plt.barh(*zip(*feature_importance))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in XGBoost')
    plt.show()


if __name__ == '__main__':
    visualize_feature_importance()