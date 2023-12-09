import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 2차원 데이터(가로: 피처, 세로: 날짜) 시계열 데이터셋

class DoubleAxisDataset(Dataset):
    pass


class DoubleAxisDataProcessor:
    def __init__(self, config):
        self.config = config
        self.config.file_paths = [config.base_dir + i for i in config.file_paths]

        self.dfs = self.import_and_list_dataframes(self.config.file_paths, self.config.use_cols)

    def get_np_data(self):
        return self.dfs

    @staticmethod
    def import_and_list_dataframes(file_paths, use_cols=None):
        """
        여러 개의 데이터프레임을 읽어오고 리스트로 반환하는 함수

        Parameters:
        - file_paths (list): 각 데이터프레임의 파일 경로가 담긴 리스트
        - use_cols (list): 주가 데이터에 사용할 feature 리스트

        Returns:
        - list: 각 데이터프레임이 담긴 리스트
        """
        if use_cols is None:
            use_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        return np.concatenate([pd.read_csv(file_path, usecols=use_cols).to_numpy() for file_path in file_paths], axis=1)

    @staticmethod
    def load_df(filepath):
        return pd.read_csv(filepath, index_col=0)

    def outer_join_df(self):
        # 데이터프레임 리스트 생성
        # dfs = [df1, df2, df3]  # 나머지 7개 데이터프레임도 추가해야 함

        # 초기 데이터프레임을 첫 번째 데이터프레임으로 설정
        merged_df = self.dfs[0]

        # 나머지 데이터프레임과 순서대로 'axis' 열을 기준으로 병합
        for df in self.dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

        # 'axis' 열을 기준으로 오름차순 정렬
        merged_df = merged_df.sort_values(by='axis')
        return merged_df


