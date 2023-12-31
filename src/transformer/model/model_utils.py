from datetime import datetime

import pandas as pd
import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib import font_manager, rc


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_stock_model(config, epoch, scaler, model, optimizer, train_losses, test_losses):
    curr_time = datetime.now().strftime("%y%m%d%H%M")
    save_path = f'{config.model_base_dir}model_state_dict_epoch_{epoch + 1}_{curr_time}.pt'
    torch.save({
        'epoch': epoch,
        'scaler': scaler,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses
    }, save_path)
    print(f"saved model: {save_path}")


def load_stock_model(load_path, model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # 저장된 파일 불러오기 예제
    checkpoint = torch.load(load_path)

    # 모델에 state_dict 적용
    model.load_state_dict(checkpoint['model_state_dict'])

    # 옵티마이저에 state_dict 적용
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 추가적인 정보 (예를 들어 손실) 불러오기
    curr_epoch = checkpoint['epoch']
    curr_train_losses = checkpoint['train_losses']
    curr_test_losses = checkpoint['test_losses']
    curr_scaler = checkpoint['scaler']

    return model, optimizer, curr_epoch, curr_train_losses, curr_test_losses, curr_scaler


def vis_losses_accs(train_losses, test_losses, config):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_losses, label='Train Loss', color='red')
    ax.plot(test_losses, label='Test Loss', color='green')

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.tick_params(labelsize=10)
    ax.legend()

    fig.suptitle("CrossAttentionTransformer Loss by Epoch", fontsize=16)
    fig.tight_layout()
    curr_time = datetime.now().strftime("%y%m%d%H%M")
    save_path = f'{config.vis_base_dir}vis_loss_{curr_time}.png'

    plt.savefig(save_path)


# 모델 예측 시각화
def vis_close_price(all_pred, n_train, n_test, targets, config):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # plt.plot(range(246), real[:246], label="real")
    # plt.plot(range(246 - 30, 246), result, label="predict")

    # 날짜 맞추기
    # np.nan을 추가할 행의 개수
    num_rows_to_add = targets.shape[0] - all_pred.shape[0]

    # np.nan으로 이루어진 배열 생성
    nan_rows = np.full((num_rows_to_add, all_pred.shape[1]), np.nan)

    # np.nan을 추가하여 새로운 배열 생성
    pred2 = np.vstack((nan_rows, all_pred))

    # 전체 ticker 기준으로 순회하면서 시각화 및 저장
    for i in range(targets.shape[1]):
        pred = pred2[:, i]
        target = targets[:, i]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(pred, label='Actual', color='red')
        ax.plot(target, label='Predict', color='blue')

        ax.set_xlabel("Time", fontsize=15)
        ax.set_ylabel("Close", fontsize=15)
        ax.tick_params(labelsize=10)
        ax.legend()

        # curr_time = datetime.now().strftime("%y%m%d%H%M")
        file_path = config.file_paths[i]

        # 정규식 패턴
        pattern = re.compile(r'final_entry/(.*?)\.csv')

        # 정규식과 매치되는 부분 찾기
        match = re.search(pattern, file_path)
        name = match.group(1)

        fig.suptitle(f"Close Price Prediction {name}", fontsize=16)
        fig.tight_layout()
        save_path = f'{config.vis_base_dir}close_{name}.png'
        plt.savefig(save_path)

        # pred_taret 별도 저장
        curr_df = pd.DataFrame({'pred': pred, 'target': target})
        curr_df.to_csv(f'{config.vis_base_dir}pred_{name}.csv', index=False)
        plt.close()
        # plt.show()

    return pred2
