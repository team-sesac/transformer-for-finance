from utils import *

class Cluster(object):
    def __init__(self, df, features=['Name', 'Close']):
        self.df = filter_data(df)

        # self.features = ['Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']
        self.features = features
        self.df = self.df[self.features]

        # 차분 및 평균, 표준편차
        pct = get_pct(self.df)
        self.data = get_mean_and_std(pct)

        # method : cosine, pca, dtw?
        self.cluster_method = kmeans_by()

    def __call__(self, method='pca', n_recent_days_list=[], n_clusters=5):
        scaled_df = pd.DataFrame()
        if len(n_recent_days_list)<2:
            scaled_df = standard_scaler(self.data)
            scaled_df_labels, reduced_data = self.cluster_method(method=method, data=scaled_df, n_clusters=n_clusters)
        else:
            reduced_data = pd.DataFrame() # pca (x,y) dataset
            cluster_lables = pd.DataFrame() # cluster lables 정보만 담길 dataset
            for days in n_recent_days_list:
                scaled_df = get_recent_and_window_data(self.data, n_recent_days=days)
                scaled_df = standard_scaler(scaled_df)
                scaled_df_labels, reduced_data = self.cluster_method(method=method, data=scaled_df, n_clusters=n_clusters)
                cluster_lables.insert(0, f'cluster_{days}', scaled_df_labels)
            
            # method == pca
            if (len(reduced_data) != 0): scaled_df = reduced_data
            
            # 클러스터 번호 통일 및 정리
            cluster_lables = unify_cluster_order(cluster_lables)
            scaled_df_labels = filter_ambiguous_clusters(cluster_lables)

        # 데이서셋에 클러스터 번호 추가
        clustered_df = add_cluster_labels(scaled_df, scaled_df_labels)
        
        # scatter 회사명 index
        clustered_df.insert(0, 'Name', self.data.index)
        clustered_df.set_index(['Name'], inplace=True)
        
        # 클러스터 == None: 삭제
        clustered_df = clustered_df.dropna()
        # 소수 클러스터 삭제
        clustered_df = filter_clusters(clustered_df)

        # 시각화
        if (method=='cosine') and (len(self.features)>2):
            print('to visualize_clusters method==\'cosine\' or features=[\'Name\', \'Close\']')
        else:
            visualize_clusters(clustered_df, clustered_df['Cluster'])

        # 라벨링된 데이터셋 반환
        return clustered_df
    
# 필히 개선 필요
class kmeans_by():
    def __init__(self):
        pass

    def __call__(self, method, data, n_clusters):
        if method == 'cosine':
            return cosine_kmeans_clustering(data=data, n_clusters=n_clusters)
        else:
            return pca_kmeans_clustering(data=data, n_clusters=n_clusters)

            

if __name__ == "__main__":
    full_file = '../krx_full_list_2016.csv'
    file = '../../data/krx_2016.csv'
    file_path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(file_path, index_col=0)
    # df = pd.read_csv(full_file, index_col=0)
    
    
    full_features = ['Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    features = ['Name', 'Close', 'Volume']
    # cluster = Cluster(df, features=full_features)
    # cluster = Cluster(df, features=features)
    cluster = Cluster(df)
    cluster_labels = cluster(method='pca', n_recent_days_list=[10, 20, 30], n_clusters=10)
    cluster_labels = cluster(method='cosine', n_recent_days_list=[10, 20, 30], n_clusters=10)
    # cluster_labels = cluster(method='pca', n_recent_days_list=None, n_clusters=10)
    # cluster_labels = cluster(method='cosine', n_recent_days_list=[10, 20, 30], n_clusters=5)
    print(cluster_labels)
    # df.to_csv(file_path)