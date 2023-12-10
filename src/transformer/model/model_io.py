from datetime import datetime
import torch
import os
import torch.optim as optim



def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_stock_model(config, epoch, model, optimizer, losses):
    curr_time = datetime.now().strftime("%y%m%d%H%M%S")
    save_path = f'{config.model_base_dir}model_state_dict_epoch_{epoch + 1}_{curr_time}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses
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
    curr_losses = checkpoint['loss']

    return model, optimizer, curr_epoch, curr_losses


