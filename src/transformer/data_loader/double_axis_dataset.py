import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 2차원 데이터(가로: 피처, 세로: 날짜) 시계열 데이터셋


class DoubleAxisDataProcessor:
    def __init__(self, config):
        self.config = config
        self.config.file_paths = [config.base_dir + i for i in config.file_paths]

        self.np_raw = self.import_and_list_dataframes(self.config.file_paths, self.config.use_cols)
        self.np_raw[np.isnan(self.np_raw)] = 0

        self.dfs = self.np_raw

        # 피처와, 타겟에 대해 Min-Max 스케일링 수행
        self.scaler = MinMaxScaler()
        # self.x_scaler = MinMaxScaler()
        # self.dfs[:, config.cols_to_scale] = self.x_scaler.fit_transform(self.dfs[:, config.cols_to_scale])
        # self.y_scaler = MinMaxScaler()
        # self.dfs[:, config.label_columns] = self.y_scaler.fit_transform(self.dfs[:, config.label_columns])

        self.dfs = self.scaler.fit_transform(self.np_raw)

    def get_np_data(self):
        return self.dfs

    def get_scaler(self):
        return self.scaler

    def split_train_test(self, test_size=0.3):
        data = self.get_np_data()
        n_train = round((1 - test_size) * len(data))
        train_data = data[:n_train]
        test_data = data[n_train:]
        return train_data, test_data

    @staticmethod
    def import_and_list_dataframes(file_paths, use_cols=None):
        """
        여러 개의 데이터프레임을 읽어오고 리스트로 반환하는 함수

        Parameters:
        - file_paths (list): 각 데이터프레임의 파일 경로가 담긴 리스트
        - use_cols (list): 주가 데이터에 사용할 feature 리스트

        Returns:
        - ndarray: 각 데이터프레임이 담긴 리스트
        """
        if use_cols is None:
            use_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        return np.concatenate([pd.read_csv(file_path, usecols=use_cols).to_numpy() for file_path in file_paths], axis=1)

    @staticmethod
    def load_df(filepath):
        return pd.read_csv(filepath, index_col=0)




