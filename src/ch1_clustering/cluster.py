from utils import *

class Cluster(object):
    def __init__(self, df):
        self.df = filter_data(df)
        self.df = self.df[['Name', 'Close']] # 개선예정

        # 차분 및 평균, 표준편차
        pct = get_pct(self.df)
        self.data = get_mean_and_std(pct)

        # 라벨 데이터셋
        self.cluster_lables = pd.DataFrame()
        self.cluster_lables.insert(0, 'Name', self.data.index)
        self.cluster_lables.set_index(['Name'], inplace=True)
        self.scaled_df = pd.DataFrame()

        self.cluster_method = kmeans_by()

    def __call__(self, method='cosine', n_recent_days_list=None, n_clusters=5):
        if n_recent_days_list==None:
            self.scaled_df = standard_scaler(self.data)
            scaled_df_labels = self.cluster_method(method=method, data=self.scaled_df, n_clusters=n_clusters)
            self.scaled_df = add_cluster_labels(self.scaled_df, scaled_df_labels)
        
        else:
            for days in n_recent_days_list:
                self.scaled_df = get_recent_and_window_data(self.data, n_recent_days=days)
                self.scaled_df = standard_scaler(self.scaled_df)
                scaled_df_labels = self.cluster_method(method=method, data=self.scaled_df, n_clusters=n_clusters)
                self.cluster_lables.insert(0, f'cluster_{days}', scaled_df_labels)

            # 클러스터 번호 통일 및 정리
            self.cluster_lables = unify_cluster_order(self.cluster_lables)
            self.cluster_lables = filter_unambiguous_clusters(self.cluster_lables)

            # 데이서셋에 클러스터 번호 추가
            self.scaled_df = add_cluster_labels(self.scaled_df, self.cluster_lables)
            self.scaled_df = self.scaled_df.dropna() # 클러스터번호가 None 삭제

        # 소수 클러스터 삭제
        self.scaled_df = filter_clusters(self.scaled_df)

        # 시각화
        visualize_clusters(self.scaled_df, self.scaled_df['Cluster'])
        # 라벨링된 데이터셋 반환
        return self.scaled_df
    
class kmeans_by():
    def __init__(self):
        pass

    def __call__(self, method, data, n_clusters):
        if method == 'cosine':
            return cosine_kmeans_clustering(data=data, n_clusters=n_clusters)
        elif method == 'pca':
            return pca_kmeans_clustering(data=data, n_clusters=n_clusters)
            

if __name__ == "__main__":
    file = '../../data/krx_2016.csv'
    file_path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(file_path, index_col=0)
    
    
    cluster = Cluster(df)
    cluster_labels = cluster(method='pca', n_recent_days_list=[10, 20, 30], n_clusters=5)
    # cluster_labels = cluster(method='cosine', n_recent_days_list=[10, 20, 30], n_clusters=5)
    print(cluster_labels)
    # df.to_csv(file_path)