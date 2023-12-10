import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.transformer.model.model_io import save_stock_model


# 훈련 함수 정의
def train(config, model, dataloader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    model.to(config.device)

    losses = list()

    for epoch in range(config.epochs):
        total_loss = 0.0
        for inputs, targets in (pbar := tqdm(dataloader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_description("%f" % (loss.item()))
            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        print(f"\n epoch: {epoch}    loss: {epoch_loss}")
        losses.append(epoch_loss)

        # 50 epoch 마다 모델의 state_dict 저장
        if (epoch + 1) % config.save_every == 0:
            save_stock_model(config, epoch, model, optimizer, losses)

    # 학습 종료 후 모델의 state_dict
    save_stock_model(config, config.epochs, model, optimizer, losses)

    return model, losses


# 검증 함수 정의
def evaluate(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


