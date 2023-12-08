import os
import ta
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


#  차트 설정
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.family"] = 'AppleGothic'
plt.rcParams["figure.figsize"] = (14,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["axes.grid"] = True

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
def get_dataset(data_list, start_date, end_date):
    # data_list = get_KRX_list()
    all_stocks = pd.DataFrame()
    for code, name in zip(data_list['Code'], data_list['Name']):
        stock = get_stock_data(code, name, start_date, end_date)
        # stock = add_full_ta(stock)
        all_stocks = pd.concat([all_stocks, stock], ignore_index=True)
    # 데이터 계층화를 위한 Date를 index로 작업
    all_stocks.set_index(['Date'], inplace=True)
    all_stocks.index.name = None
    return all_stocks

# 테스트용 50개 회사만 조회
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

# 각 회사의 전체 주식 거래 기간의 30% 이상이 Volume이 0인 경우 해당 회사 데이터 삭제
def _filter_zero_volume_data(df):
    volume_zero_percentage = df.groupby('Name')['Volume'].apply(lambda x: (x == 0).mean())
    companies_to_remove = volume_zero_percentage[volume_zero_percentage > 0.3].index
    # 조건에 맞는 회사 데이터 삭제
    df_filtered = df[~df['Name'].isin(companies_to_remove)]
    return df_filtered

def filter_data(df):
    # 전처리
    df = _filter_zero_volume_data(df)
    feature_columns = df.columns.difference(['Code']) # Code 열 제거
    df = df[feature_columns] 
    
    # 결측치 처리
    df = df.replace({0: 0.000001, None: 0.000001}) 
    df = df.dropna(axis=0) # 이동평균선 등 NaN 행 제거
    # df = df.fillna(0.000001)
    return df
    
def standard_scaler(df):
    # 변동성을 퍼센트로 변경
    pct = df.groupby('Name').pct_change()
    pct['Name'] = df['Name']

    data = pd.DataFrame()
    data = pct.groupby('Name').mean()
    data = pd.concat([data, pct.groupby('Name').std()], axis=1)

    # scaler
    scaler = StandardScaler()#.fit(data)
    rescaled_dataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return rescaled_dataset

def grouped_data_standard_scaler(df):
    # 날짜 컬럼을 Datetime 형식으로 변환
    df.index = pd.to_datetime(df.index)
    
    # pct_change 계산
    pct = df.groupby('Name').pct_change()
    pct['Name'] = df['Name']
    
    # 필요한 날짜 기준으로 groupby하여 평균과 표준편차 계산
    grouped_data = pd.DataFrame()

    for days in [10, 20, 30, 180]:
        grouped_data[f'mean_{days}d'] = pct.groupby('Name').rolling(window=days).mean().reset_index(drop=True)
        grouped_data[f'std_{days}d'] = pct.groupby('Name').rolling(window=days).std().reset_index(drop=True)

    return grouped_data

# 코사인 유사도 + K-Means 클러스터링
def cosine_kmeans_clustering(data, n_clusters):
    similarity_matrix = cosine_similarity(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) # random_state : cluster 번호가 변하여 고정
    labels = kmeans.fit_predict(similarity_matrix)
    return labels

# PCA + K-Means 클러스터링
def pca_kmeans_clustering(data, n_clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42) # random_state : cluster 번호가 변하여 고정
    labels = kmeans.fit_predict(reduced_data)
    return labels

# DTW 거리 계산 함수. 미완
def dtw_distance(series1, series2):
    distance, path = fastdtw(series1, series2, dist=euclidean)
    return distance
def dtw_kmeans_clustering(data, n_clusters):
    # DTW 거리 행렬 생성
    dtw_matrix = pd.DataFrame(index=data.index, columns=data.index)
    for i in range(len(data.index)):
        for j in range(len(data.index)):
            dtw_matrix.at[i, j] = dtw_distance(data.values[i], data.values[j])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(dtw_matrix)
    return labels

# 클러스터링 결과 시각화. 개선 필요
def visualize_clusters(data, labels):
    # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis') # data: numpy
    plt.scatter(data.values[:, 0], data.values[:, 1], c=labels, cmap='viridis', label=data.index) # data: pandas
    plt.title('Cluster Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# MultiIndex에서 회사별로 클러스터 라벨 추가
def add_cluster_labels(df, labels):
    df['Cluster'] = labels
    return df

# n개 미만 클러스터 삭제. 이전 버전.
def filter_clusters(df, n=3):
    cluster_counts = df['Cluster'].value_counts()
    clusters_to_keep = cluster_counts[cluster_counts >= n].index
    df_filtered = df[df['Cluster'].isin(clusters_to_keep)]
    return df_filtered

# min_count 개 미만 클러스터 제거. 개선 버전
def filter_clusters_with_min_count(cluster_labels, min_count=3):
    for idx_cluster in range(cluster_labels.shape[1]):
        cluster_counts = cluster_labels.iloc[:,idx_cluster].value_counts()
        clusters_to_keep = cluster_counts[cluster_counts >= min_count].index
        cluster_labels = cluster_labels[cluster_labels.iloc[:,idx_cluster].isin(clusters_to_keep)]
    return cluster_labels

# main 함수 예시
def make_cluster_labels_dataset(df):
    df = filter_data(df)
    rescaled_dataset = standard_scaler(df)
    pca_labels = pca_kmeans_clustering(rescaled_dataset,5)
    cosine_labels = cosine_kmeans_clustering(rescaled_dataset,5)

    cluster_labels = pd.DataFrame()
    cluster_labels.index = rescaled_dataset.index
    cluster_labels['pca'] = pca_labels
    cluster_labels['cosine'] = cosine_label
    cluster_labels = filter_clusters_with_min_count(cluster_labels)
    return cluster_labels

# main 함수 예시. 종가만 사용
def make_cluster_labels_dataset_by_close(df):
    df = filter_data(df)
    df = df[['Name', 'Close']]
    rescaled_dataset = standard_scaler(df)
    # pca_labels = pca_kmeans_clustering(rescaled_dataset,5)
    cosine_labels = cosine_kmeans_clustering(rescaled_dataset,5)

    # cluster_labels = pd.DataFrame()
    # cluster_labels.index = rescaled_dataset.index
    # cluster_labels['pca'] = pca_labels
    # cluster_labels['cosine'] = cosine_labels
    # 소수 클러스터 삭제
    # cluster_labels = filter_clusters_with_min_count(cluster_labels)

    # 종가만 사용할 경우 시각화 작업
    # pca_cluster = add_cluster_labels(rescaled_dataset, pca_labels)
    cosine_cluster = add_cluster_labels(rescaled_dataset, cosine_labels)
    # 소수 클러스터 삭제
    # pca_cluster = filter_clusters(pca_cluster)
    cosine_cluster = filter_clusters(cosine_cluster)
    
    # visualize cluster
    # visualize_clusters(pca_cluster, pca_cluster['Cluster'])
    # visualize_clusters(cosine_cluster, cosine_cluster['Cluster'])
    
    return cosine_cluster

def make_cluster_labels_dataset_by_grouped_daata(df):
    df = filter_data(df)
    df = df[['Name', 'Close']]
    rescaled_dataset = grouped_data_standard_scaler(df)
    print(rescaled_dataset)
    cosine_labels = cosine_kmeans_clustering(rescaled_dataset,5)

    return cosine_labels



if __name__ == "__main__":
    # file = '../../data/krx_ta_2016.csv'
    file = '../../data/krx_2016.csv'
    file_path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(file_path, index_col=0)
    # start_date, end_date = '20160101', '20231206'
    # df = get_test_dataset(start_date, end_date)
    # df = get_dataset(start_date, end_date)

    cluster_labels = make_cluster_labels_dataset_by_grouped_daata(df)
    print(cluster_labels)
    # df.to_csv(file_path)