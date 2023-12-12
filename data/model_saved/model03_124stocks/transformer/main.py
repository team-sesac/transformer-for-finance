from src.transformer.data_loader.double_axis_dataset import DoubleAxisDataProcessor
from src.transformer.data_loader.single_axis_dataset import TimeSeriesDataset
from src.transformer.model.CrossAttn import CrossAttentionTransformer
from src.transformer.model.model_utils import create_directory_if_not_exists, load_stock_model, vis_losses_accs, \
    save_stock_model, vis_close_price
from src.transformer.train.train_stock import train, evaluate, predict
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
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


def load_model_and_evaluate():
    config = Config()

    # 데이터
    data_processor = DoubleAxisDataProcessor(config)
    train_data, test_data = data_processor.split_train_test(test_size=config.test_size)
    scaler = data_processor.get_scaler()
    all_data = data_processor.get_np_data()

    # 모든 데이터
    all_dataset = TimeSeriesDataset(config, all_data)
    all_dataloader = DataLoader(dataset=all_dataset, batch_size=len(all_dataset), shuffle=False, drop_last=True)

    # # 최근 20% 일자
    # test_dataset = TimeSeriesDataset(config, test_data)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_data), shuffle=False, drop_last=True)
    # 모델
    model = CrossAttentionTransformer(config).to(config.device)
    model_dir = config.model_base_dir+config.model_to_load
    model, optimizer, curr_epoch, curr_train_losses, curr_test_losses, _ = load_stock_model(model_dir, model, config)

    all_pred = predict(all_dataloader, model, config.device)
    targets = all_data[:, config.label_columns[0]::config.len_feature_columns]

    # pred_inversed = scaler.inverse_transform(all_pred)
    # target_inversed = scaler.inverse_transform(targets)
    vis_close_price(all_pred, len(train_data), len(test_data), targets, config)



def main():
    # 설정
    config = Config()
    create_directory_if_not_exists(config.model_base_dir)
    create_directory_if_not_exists(config.vis_base_dir)

    # 데이터
    data_processor = DoubleAxisDataProcessor(config)
    train_data, test_data = data_processor.split_train_test(test_size=config.test_size)
    scaler = data_processor.get_scaler()
    train_dataset = TimeSeriesDataset(config, train_data)
    test_dataset = TimeSeriesDataset(config, test_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_train_data, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True)

    # 모델 학습 및 저장
    model = CrossAttentionTransformer(config)
    train_losses, test_losses = list(), list()

    # pretrained model 로드해 학습하는 경우
    if config.do_continue_train:
        model_dir = config.model_base_dir + config.model_to_load
        model, optimizer, curr_epoch, train_losses, test_losses = load_stock_model(model_dir, model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    model.to(config.device)

    for epoch in range(config.epochs):
        model.train()
        train_epoch_loss = train(train_dataloader, model, criterion, optimizer, config.device)
        train_losses.append(train_epoch_loss)

        model.eval()
        test_epoch_loss = evaluate(test_dataloader, model, criterion, config.device)
        test_losses.append(test_epoch_loss)
        print(f"epoch: {epoch} train_loss: {train_epoch_loss:.6f} test_loss: {test_epoch_loss:.6f}")

        # 50 epoch 마다 모델의 state_dict 저장
        if (epoch + 1) % config.save_every == 0:
            save_stock_model(config, config.epochs, scaler, model, optimizer, train_losses, test_losses)

        # 학습 종료 후 모델의 state_dict
    save_stock_model(config, config.epochs, scaler, model, optimizer, train_losses, test_losses)

    vis_losses_accs(train_losses, test_losses, config)

    model.eval()
    result = predict(dataloader=test_dataloader, model=model, device=config.device)


    # 평가 및 시각화
    print('here')


if __name__ == '__main__':
    # test()
    main()
    # load_model_and_evaluate()
    # import random
    # config = Config()
    # create_directory_if_not_exists(config.vis_base_dir)
    #
    # train_losses = [random.random() for _ in range(10)]
    # test_losses = [random.random() for _ in range(10)]
    #
    # # 예제 실행
    # vis_losses_accs(train_losses, test_losses, config)

