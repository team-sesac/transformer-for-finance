import os
import ta
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib


#  차트 설정
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.family"] = 'AppleGothic'
# plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings(action='ignore')

random_state = 42

# 한국거래소 상장종목 전체 조회
def get_KRX_list():
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    kospi_list = kospi[['Code', 'Name']]
    kosdaq_list = kosdaq[['Code', 'Name']]
    data_list = pd.concat([kospi_list, kosdaq_list], axis=0)
    return data_list

# Close 데이터셋
def get_close_data(stock_code, stock_name, start_date, end_date):
    df = fdr.DataReader(stock_code,start_date, end_date)
    df.columns = stock_name

    # 전처리
    # 30% 이상 거래 없는 종목 제거
    missing_fractions = df.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    # 결측치를 바로 전일과 동일하게 설정(ffill)
    df.drop(labels=drop_list, axis=1, inplace=True)
    close_df = df.fillna(method='ffill')
    return close_df

# 평균수익, 변동성
def get_pct(df):
    days = df.shape[0]
    returns = df.pct_change().mean() * days
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = df.pct_change().std() * np.sqrt(days)
    return returns

# 스케일러
def standard_scaler(df):
    scaler = StandardScaler().fit(df)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df

# 이상치 제거
def remove_outliers(df):
    """
    DataFrame에서 이상치를 제거하는 함수.

    Parameters:
    - df (pd.DataFrame): 이상치를 제거할 DataFrame.

    Returns:
    - pd.DataFrame: 이상치가 제거된 DataFrame.
    """
    q3 = df.quantile(0.75)
    q1 = df.quantile(0.25)
    iqr = q3 - q1

    condition = (df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))
    condition = condition.any(axis=1)
    search_df = df[condition]

    df = df.drop(search_df.index, axis=0)
    return df

# Affinity Propagation Clustering
def affinity_clustering(df, damping=0.5, max_iter=250):
    ap = AffinityPropagation(damping=damping, max_iter=max_iter, affinity='euclidean')
    ap.fit(df)
    labels = ap.predict(df)
    return labels

# 최적의 클러스터 수 찾기
def find_optimal_clusters(df, max_clusters=10):
    min_silhouette_score = 1.0
    n_clusters = 4  # 초기 클러스터 수
    for k in range(n_clusters, max_clusters):
        kmeans_test = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans_test.fit(df)
        silhouette_score = metrics.silhouette_score(df, kmeans_test.labels_, random_state=random_state)

        if silhouette_score < min_silhouette_score:
            min_silhouette_score = silhouette_score
        else:
            n_clusters = k
            break

    return n_clusters

# 코사인 유사도 + K-Means 클러스터링
def cosine_kmeans_clustering(df, n_clusters=10):
    similarity_matrix = cosine_similarity(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state) # random_state : cluster 번호가 변하여 고정
    labels = kmeans.fit_predict(similarity_matrix)
    centroids = kmeans.cluster_centers_ # 각 클러스터 중점
    return labels, centroids

def get_labels_optimal_cluster(df, max_clusters=10):
    """
    주어진 데이터에 대해 최적의 클러스터 수로 cosine Kmeans Clustering 함수.

    Parameters:
    - df (pd.DataFrame): 클러스터링을 수행할 데이터.
    - max_clusters (int): 시도할 최대 클러스터 수.

    Returns:
    - ndarry: cluster labels
    """
    n_clusters = find_optimal_clusters(df, max_clusters=max_clusters)
    labels, centroids = cosine_kmeans_clustering(df, n_clusters=n_clusters)

    # 하이리턴 하이리스트 라벨
    hight_return_volatility_index = (centroids[:,0]> 0.0) & (centroids[:,1] > 0.0)
    hight_return_volatility =  [index for index, value in enumerate(hight_return_volatility_index) if value]
    return labels, hight_return_volatility

#  클러스터 Name, Code 추가
def add_cluster_info(df, stock_list):
    df['Name'] = stock_list['Name'].to_numpy()
    df['Code'] = stock_list['Code'].to_numpy()
    return df

#  클러스터 라벨 추가
def add_cluster_labels(df, labels):
    df['Cluster'] = labels
    return df

# 클러스터링 결과 시각화
def visualize_clusters(df):
    fig, ax = plt.subplots()
    sc = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2], cmap='rainbow')
    plt.title('Cluster')
    plt.xlabel('Mean Return')
    plt.ylabel('Volatility')

    # legend : Cluster labels 
    ax.legend(*sc.legend_elements(), title='clusters')
    # plt.colorbar(sc) # 오르쪽 컬러바

    # n개 미만일 경우 회사명 명시
    # if df.shape[0] < 100:
    #     for x, y, name in zip(df.iloc[:, 0], df.iloc[:, 1], df.index):
    #         label = name
    #         plt.annotate(label, # this is the text
    #                     (x,y), # this is the point to label
    #                     textcoords="offset points", # how to position the text
    #                     xytext=(0,10), # distance from text to points (x,y)
    #                     ha='center', # horizontal alignment can be left, right or center
    #                     alpha=0.5) 
    plt.show()


def get_groupby_days_data(df, n_recent_days=None):
    result = []
    if n_recent_days is not None:
        for group_name, group_data in df.groupby('Name'):
            recent_data = group_data.tail(n_recent_days)
            result.append(recent_data)
    return pd.concat(result, ignore_index=True)



# 클러스터 번호 통일
def unify_cluster_order(df):
    for col in df.columns:
        unique_values = df[col].unique()
        mapping_dict = {value: idx for idx, value in enumerate(unique_values)}
        df[col] = df[col].map(mapping_dict)
    return df

# 클러스터 번호가 다른 행 None 변경.
def filter_ambiguous_clusters(df):
    df[~df.duplicated(keep=False)] = None
    return df.values[:,0]

# n개 미만 클러스터 삭제
def filter_clusters(df, min_count=3):
    cluster_counts = df['Cluster'].value_counts()
    clusters_to_keep = cluster_counts[cluster_counts>=min_count].index
    df_filtered = df[df['Cluster'].isin(clusters_to_keep)]
    return df_filtered

# main 함수 예시
def get_cluster_labels_dataset(stock_list, start_date, end_date):
    # 종가 데이터 추출
    # start_date, end_date = '20231101', '20231130'
    # stock_list = get_KRX_list()[:50]

    # 데이터 가공
    df = get_close_data(stock_list['Code'], stock_list['Name'], start_date, end_date)
    df = get_pct(df)
    df = standard_scaler(df)
    df = remove_outliers(df)

    # cosine clustering
    labels, hight_return_volatility = get_labels_optimal_cluster(df)

    # 가공된 데이터셋에 클러스터 라벨 병합
    df = add_cluster_labels(df, labels)
    return df




if __name__ == "__main__":
    # file = '../../data/krx_ta_2016.csv'
    # file = '../../data/krx_2016.csv'
    # file_path = os.path.join(os.path.dirname(__file__), file)
    # df = pd.read_csv(file_path, index_col=0)
    
    # 설정
    start_date, end_date = '20231101', '20231130'
    stock_list = get_KRX_list()[:50]

    # 실행
    df = get_cluster_labels_dataset(stock_list, start_date, end_date)
    # print(df)

    # 시각화
    visualize_clusters(df, df['Cluster'].to_numpy())
    # df.to_csv(file_path)