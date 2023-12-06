import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import os
from itertools import cycle 

# utils.py
from utils import *

# 차트 설정
# %matplotlib inline
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.family"] = 'AppleGothic'
plt.rcParams["figure.figsize"] = (14, 4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["axes.grid"] = True

def read_stock_data(file_path):
    origin_df = pd.read_csv(file_path)
    return origin_df

def preprocess_data(df):
    # 거래량 없는 데이터 삭제
    df = df[df['Volume'] != 0]

    # 종가 데이터만 추출
    df = df[['Name', 'Close']]
    return df

def calculate_returns_volatility(df):
    days = df.groupby(['Name']).size().values.reshape(-1, 1)
    pct = pd.DataFrame(df.groupby('Name').pct_change())
    pct['Name'] = df['Name']
    returns = pct.groupby('Name').mean() * days
    returns.columns = ['Returns']
    returns['Volatility'] = pct.groupby('Name').std() * np.sqrt(days)
    return returns

def scale_data(data):
    scaler = StandardScaler().fit(data)
    rescaled_dataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return rescaled_dataset

def plot_affinity_clusters(X, labels, cluster_centers_indices):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    axes = axes.flatten()
    scatter = axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="rainbow")
    axes[0].set_title('Affinity')
    axes[0].set_xlabel('Mean Return')
    axes[0].set_ylabel('Volatility')

    for x, y, name in zip(X.iloc[:, 0], X.iloc[:, 1], X.index):
        label = name
        axes[0].annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)

    X_temp = np.asarray(X)


    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X_temp[cluster_centers_indices[k]]
        axes[1].plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
        axes[1].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X_temp[class_members]:
            axes[1].plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    fig.colorbar(scatter)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file = '../../data/krx_2016.csv'
    file_path = os.path.join(os.path.dirname(__file__), file)
    origin_df = read_stock_data(file_path)
    df = preprocess_data(origin_df)
    returns_volatility = calculate_returns_volatility(df)
    scaled_data = scale_data(returns_volatility)

    # Affinity Propagation
    ap = AffinityPropagation(damping=0.5, max_iter=250, affinity='euclidean')
    ap.fit(scaled_data)
    clust_labels2 = ap.predict(scaled_data)

    plot_affinity_clusters(scaled_data, clust_labels2, ap.cluster_centers_indices_)
