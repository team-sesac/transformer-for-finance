import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer
from torch import nn
import torch
import math


class WindowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        # 총 데이터의 개수
        L = y.shape[0]
        # stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        # input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[:, i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, 2))
        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i, :-1], self.y[i, 1:]

    def __len__(self):
        return self.len


class TFModel(nn.Module):
    def __init__(self,d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.transformer = Transformer(d_model=d_model, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, num_decoder_layers=nlayers,dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_encoder_d = PositionalEncoding(d_model, dropout)
        self.linear = nn.Linear(d_model, 1)
        self.encoder = nn.Linear(1, d_model)
        self.encoder_d = nn.Linear(1, d_model)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, srcmask, tgtmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt)
        tgt = self.pos_encoder_d(tgt)
        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), srcmask, tgtmask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


def evaluate(length):
    input = torch.tensor(data_train[-60:]).reshape(1,-1,1).to(device).float().to(device)
    output = torch.tensor(data_train[-1].reshape(1,-1,1)).float().to(device)
    model.eval()
    for i in range(length):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        predictions = model(input, output, src_mask, tgt_mask).transpose(0,1)
        predictions = predictions[:, -1:, :]
        output = torch.cat([output, predictions.to(device)], axis=1)
    return torch.squeeze(output, axis=0).detach().cpu().numpy()[1:]

#####
df = fdr.DataReader('KS11', '2022-01-01', '2022-12-31')

min_max_scaler = MinMaxScaler()
df['Close'] = min_max_scaler.fit_transform(df['Close'].to_numpy().reshape(-1,1))

train = df[:-30]
data_train = train['Close'].to_numpy()

test = df[-30:]
data_test = test['Close'].to_numpy()

iw = 60
ow = 30

train_dataset = WindowDataset(data_train, input_window=iw, output_window=ow, stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)

device = torch.device("cuda")

lr = 1e-3
model = TFModel(256, 8, 256, 2, 0.1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epoch = 2000
from tqdm import tqdm

model.train()
progress = tqdm(range(epoch))
for i in progress:
batchloss = 0.0

for (inputs, dec_inputs, outputs) in train_loader:
    optimizer.zero_grad()
    src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
    tgt_mask = model.generate_square_subsequent_mask(dec_inputs.shape[1]).to(device)

    result = model(inputs.float().to(device), dec_inputs.float().to(device), src_mask, tgt_mask)
    loss = criterion(result.permute(1,0,2), outputs.float().to(device))

    loss.backward()
    optimizer.step()
    batchloss += loss
progress.set_description("{:0.5f}".format(batchloss.cpu().item() / len(train_loader)))


result = evaluate(30)
result = min_max_scaler.inverse_transform(result)
real = df['Close'].to_numpy()
real = min_max_scaler.inverse_transform(real.reshape(-1,1))