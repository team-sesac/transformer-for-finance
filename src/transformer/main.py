from src.transformer.data_loader.double_axis_dataset import DoubleAxisDataProcessor
from src.transformer.data_loader.single_axis_dataset import TimeSeriesDataset
from src.transformer.model.CrossAttn import CrossAttnModel, CrossAttnModel2
from src.transformer.model.model_io import create_directory_if_not_exists, load_stock_model
from src.transformer.train.train_stock import train, evaluate
from torch.utils.data import DataLoader
import torch.nn as nn
from config import Config


def test():
    import numpy as np

    # 주어진 정보
    y_row = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    len_feature_columns = 4
    label_columns = [0, 2]

    # 한 줄 for문으로 처리
    result = np.array([y_row[i:i + len_feature_columns][label_columns] for i in range(0, len(y_row), len_feature_columns)]).flatten()

    # 결과 출력
    print(result)


def load_model_test():
    config = Config()

    # 데이터
    data_processor = DoubleAxisDataProcessor(config)
    np_stock_data = data_processor.get_np_data()
    dataset = TimeSeriesDataset(config, np_stock_data)
    test_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    # 모델
    model = CrossAttnModel2(config).to(config.device)
    model_dir = config.model_base_dir+"model_state_dict_epoch_3_20231210121644.pt"
    model, optimizer, curr_epoch, curr_losses = load_stock_model(model_dir, model, config)
    criterion = nn.MSELoss()
    eval_loss = evaluate(model, test_loader, criterion, config)
    print(eval_loss)

def main():
    # 설정
    config = Config()
    create_directory_if_not_exists(config.model_base_dir)

    # 데이터
    data_processor = DoubleAxisDataProcessor(config)
    np_stock_data = data_processor.get_np_data()
    dataset = TimeSeriesDataset(config, np_stock_data)
    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    # 모델 학습 및 저장
    # model = CrossAttnModel(config.seq_len, config.input_size, config.output_size, config.heads)
    model = CrossAttnModel2(config)
    model, losses = train(config, model, train_loader)

    # 평가 및 시각화

    print('here')


if __name__ == '__main__':
    # test()
    # main()
    load_model_test()

