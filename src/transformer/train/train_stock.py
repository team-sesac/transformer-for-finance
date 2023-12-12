import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.transformer.model.model_utils import save_stock_model


# 훈련 함수 정의

def train(dataloader, model, criterion, optimizer, device):
    epoch_loss = 0.
    for X, y in (pbar := tqdm(dataloader)):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"{loss.item():.4f} {pred[-1, 0].detach().cpu():3.3f} {y[-1, 0].detach().cpu():3.3f}" % ())
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    return epoch_loss


# 검증 함수 정의
def evaluate(dataloader, model, criterion, device):
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# 예측 함수 정의
def predict(dataloader, model, device):
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            return predictions.detach().cpu().numpy()



#####################
def train_legacy(config, model, train_dataloader, test_dataloader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    model.to(config.device)

    train_losses = list()
    test_losses = list()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in (pbar := tqdm(train_dataloader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_description("%f" % (loss.item()))
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_dataloader)
        print(f"\n epoch: {epoch}    loss: {epoch_loss}")
        train_losses.append(epoch_loss)

        # test_dataloader validation
        epoch_test_loss = evaluate(model, test_dataloader, criterion, config)

        # 50 epoch 마다 모델의 state_dict 저장
        if (epoch + 1) % config.save_every == 0:
            save_stock_model(config, epoch, model, optimizer, train_losses)

    # 학습 종료 후 모델의 state_dict
    test_loss = save_stock_model(config, config.epochs, model, optimizer, train_losses)

    return model, train_losses

