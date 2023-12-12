from datetime import datetime
import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_stock_model(config, epoch, model, optimizer, train_losses, test_losses):
    curr_time = datetime.now().strftime("%y%m%d%H%M")
    save_path = f'{config.model_base_dir}model_state_dict_epoch_{epoch + 1}_{curr_time}.pt'
    torch.save({
        'epoch': epoch,
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

    return model, optimizer, curr_epoch, curr_train_losses, curr_test_losses


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
