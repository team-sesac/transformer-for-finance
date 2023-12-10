import torch
from numpy.distutils.command.config import config
from torch import nn
import transformers
from performer_pytorch import Performer

class TestModel(nn.Module):
    def __init__(self, seq, input_size, output_size, nhead):
        super(TestModel, self).__init__()

        self.linear_query1 = nn.Linear(input_size, output_size)
        self.linear_key1 = nn.Linear(input_size, output_size)
        self.linear_value1 = nn.Linear(input_size, output_size)
        self.attention1 = nn.MultiheadAttention(embed_dim=input_size, num_heads=nhead)

        self.linear_query2 = nn.Linear(input_size, output_size)
        self.linear_key2 = nn.Linear(input_size, output_size)
        self.linear_value2 = nn.Linear(input_size, output_size)
        self.attention2 = nn.MultiheadAttention(embed_dim=input_size, num_heads=nhead)

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        query1 = self.linear_query1(x)
        key1 = self.linear_key1(x)
        value1 = self.linear_value1(x)

        # 첫 번째 어텐션 (세로 방향)
        x1, _ = self.attention2(query1.transpose(0, 1), key1.transpose(0, 1), value1.transpose(0, 1))
        x1 = x1.transpose(0, 1)

        query2 = self.linear_query2(x1)
        key2 = self.linear_key2(x1)
        value2 = self.linear_value2(x1)

        # 두 번째 어텐션 (가로 방향)
        x2, _ = self.attention1(query2, key2, value2)

        # 평균 풀링
        x2 = torch.mean(x2, dim=1)

        # Fully Connected Layer
        x2 = self.fc(x2)

        return x2


class CrossAttnModel(nn.Module):
    def __init__(self, seq, input_size, output_size, nhead):
        super(CrossAttnModel, self).__init__()

        self.linear_query1 = nn.Linear(input_size, input_size)
        self.linear_key1 = nn.Linear(input_size, input_size)
        self.linear_value1 = nn.Linear(input_size, input_size)
        self.attention1 = nn.MultiheadAttention(embed_dim=input_size, num_heads=nhead)

        self.linear_query2 = nn.Linear(input_size, input_size)
        self.linear_key2 = nn.Linear(input_size, input_size)
        self.linear_value2 = nn.Linear(input_size, input_size)
        self.attention2 = nn.MultiheadAttention(embed_dim=input_size, num_heads=nhead)

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        query1 = self.linear_query1(x)
        key1 = self.linear_key1(x)
        value1 = self.linear_value1(x)

        # 첫 번째 어텐션 (세로 방향)
        x1, _ = self.attention2(query1.transpose(0, 1), key1.transpose(0, 1), value1.transpose(0, 1))
        x1 = x1.transpose(0, 1)

        query2 = self.linear_query2(x1)
        key2 = self.linear_key2(x1)
        value2 = self.linear_value2(x1)

        # 두 번째 어텐션 (가로 방향)
        x2, _ = self.attention1(query2, key2, value2)

        # 평균 풀링
        x2 = torch.mean(x2, dim=1)

        # Fully Connected Layer
        x2 = self.fc(x2)

        return x2


######################################################################

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).cuda()  # CUDA로 이동
        position = torch.arange(0, max_len).unsqueeze(1).float().cuda()  # CUDA로 이동
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).cuda()  # CUDA로 이동

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


'''
메인 모델 v1
'''

class CrossAttnModel2(nn.Module):
    def __init__(self, config):
        super(CrossAttnModel2, self).__init__()
        self.config = config

        # 리니어 레이어를 담을 리스트
        # self.linear_layers = nn.ModuleList([nn.Linear(4, 1) for _ in range(10)])
        self.linear_layers = nn.ModuleList([nn.Linear(in_features=config.len_feature_columns, out_features=config.n_labels, dtype=torch.float32).to(config.device) for _ in range(config.n_files)])

        # BERT 레이어
        # self.bert = nn.Linear(10, 3)  # 입력 10, 출력 3
        # self.bert = nn.Linear(in_features=config.n_files, out_features=config.n_labels)  # 입력:종목, 출력:예측할 가격들
        # self.time_transformer = transformers.TimeSeriesTransformerModel(
        #     config=transformers.TimeSeriesTransformerConfig(
        #         prediction_length=1,
        #         hidden_size=768,
        #         num_hidden_layers=3,
        #         num_attention_heads=3,
        #         intermediate_size=3072,
        #         dropout=0.1,
        #         attention_dropout=0.1,
        #         activation="gelu",
        #     ),
        # )

        # https://github.com/lucidrains/performer-pytorch
        self.performer_ticker = Performer(
            dim=config.seq_len,
            depth=4,
            heads=int(config.n_files/2),
            dim_head=4,
            causal=False
        ).to(config.device)

        self.performer_time = Performer(
            dim=config.n_files,
            depth=4,
            heads=int(config.n_files/2),
            dim_head=4,
            causal=False
        ).to(config.device)

        self.pos_embedding_ticker = PositionalEmbedding(config.n_files, config.seq_len)
        self.pos_embedding_time = PositionalEmbedding(config.seq_len, config.n_files)

        # LSTM 레이어
        # self.lstm = nn.LSTM(input_size=30, hidden_size=10, batch_first=True)
        self.lstm = nn.LSTM(input_size=config.n_labels, hidden_size=config.n_files, batch_first=True, dtype=torch.float32)

    def forward(self, x):
        # 리니어 레이어를 통과하여 결과를 리스트에 저장
        # linear_outputs = [linear(x[:, i * 4:(i + 1) * 4]) for i, linear in enumerate(self.linear_layers)]

        linear_outputs = None
        for i, linear in enumerate(self.linear_layers):
            # (Batch, seq_len, len_feature_columns)
            y_start, y_end = 0, self.config.seq_len
            x_start, x_end = i * self.config.len_feature_columns, (i + 1) * self.config.len_feature_columns
            curr_seq_feats = x[:, y_start:y_end, x_start:x_end]
            curr_output = linear(curr_seq_feats)
            if linear_outputs is None:
                linear_outputs = curr_output
                continue

            linear_outputs = torch.cat([linear_outputs, curr_output], dim=2)

        # linear_outputs = [linear(x[:, i * self.config.len_feature_columns:(i + 1) * self.config.len_feature_columns]) for i, linear in enumerate(self.linear_layers)]

        # 리스트를 텐서로 변환
        # linear_outputs = torch.stack(linear_outputs, dim=1)

        # time_transformer_output = self.time_transformer(linear_outputs)

        # BERT 레이어 통과
        # bert_output = self.transformer.forward(past_values=linear_outputs)

        # LSTM 레이어에 통과
        # lstm_output, _ = self.lstm(bert_output)

        # x = self.performer_ticker(linear_outputs)
        # x = self.performer_time(x)

        outputs = linear_outputs.permute(0, 2, 1)
        outputs = self.pos_embedding_ticker(outputs)
        outputs = self.performer_ticker(outputs)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.pos_embedding_time(outputs)
        outputs = self.performer_time(outputs)

        outputs = outputs.mean(dim=1)

        return outputs