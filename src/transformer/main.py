import torch

from src.transformer.data_loader.double_axis_dataset import DoubleAxisDataProcessor
from src.transformer.data_loader.single_axis_dataset import TimeSeriesDataset
from src.transformer.model.CrossAttn import CrossAttnModel, CrossAttnModel2
from src.transformer.train.train_stock import train
from torch.utils.data import DataLoader

from config import Config


def test():
    import numpy as np

    # 주어진 정보
    y_row = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    len_feature_columns = 4
    label_columns = [0, 2]

    # 한 줄 for문으로 처리
    # result = np.array([y_row[i:i + len_feature_columns][label_columns] for i in range(0, len(y_row), len_feature_columns)])
    result = np.array([y_row[i:i + len_feature_columns][label_columns] for i in range(0, len(y_row), len_feature_columns)]).flatten()

    # 결과 출력
    print(result)


def main():
    config = Config()

    data_processor = DoubleAxisDataProcessor(config)
    np_stock_data = data_processor.get_np_data()
    dataset = TimeSeriesDataset(config, np_stock_data)
    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    # model = CrossAttnModel(config.seq_len, config.input_size, config.output_size, config.heads)
    model = CrossAttnModel2(config)

    train(config, model, train_loader)

    print('here')


if __name__ == '__main__':
    # test()
    main()

