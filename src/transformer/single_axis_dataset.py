import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.window_size])
        y = torch.tensor(self.data[idx+self.window_size])
        return x, y


class TransformerDataset(Dataset):
    def __init__(self, filepath, cols, test_size):
        self.filepath = filepath
        self.data = self.get_csum_logreturn(cols=cols)
        self.train_data, self.test_data = self.split_train_test(test_size=test_size)

    def __call__(self, window_size):
        train_window = TimeSeriesDataset(self.train_data, window_size)
        test_window = TimeSeriesDataset(self.test_data, window_size)
        return train_window, test_window

    def get_csum_logreturn(self, cols):
        # 데이터 스케일링
        df = pd.read_csv(self.filepath, index_col=0)
        close = df[cols].to_numpy().flatten()
        logreturn = np.diff(np.log(close))
        csum_logreturn = logreturn.cumsum()  # 누적 수익률
        return csum_logreturn

    def split_train_test(self, test_size=0.3):
        n_train = round((1 - test_size) * len(self.data))
        train_data = self.data[:n_train]
        test_data = self.data[n_train:]
        return train_data, test_data


if __name__ == '__main__':
    filepath = '../../data/tf_dataset/186_SK케미칼_2010.csv'
    cols = ['Close']
    test_size = 0.3
    trans_dataset = TransformerDataset(filepath, cols, test_size)
    train_dataset, test_dataset = trans_dataset(window_size=10)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # 데이터 확인
    for x, y in dataloader:
        print("Input (Sliding Window):", x.view(-1).numpy())
        print("Target (Next Value):", y.item())
        print()

    print('here')