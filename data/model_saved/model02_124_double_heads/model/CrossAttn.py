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

class CrossAttentionTransformer(nn.Module):
    def __init__(self, config):
        super(CrossAttentionTransformer, self).__init__()
        self.config = config

        # 리니어 레이어를 담을 리스트
        # self.linear_layers = nn.ModuleList([nn.Linear(4, 1) for _ in range(10)])
        self.linear_layers = nn.ModuleList([nn.Linear(in_features=config.len_feature_columns, out_features=config.n_labels, dtype=torch.float32).to(config.device) for _ in range(config.n_files)])

        self.pos_embedding_ticker = PositionalEmbedding(config.n_files, config.seq_len)

        # https://github.com/lucidrains/performer-pytorch
        self.performer_ticker = Performer(
            dim=config.seq_len, # 트랜스포머의 입력 임베딩 및 출력 차원입니다. 이는 시퀀스의 길이 또는 특성의 차원을 나타냅니다.
            depth=4,            # 트랜스포머의 레이어 수입니다. 이는 트랜스포머가 몇 개의 층으로 구성되어 있는지를 나타냅니다.
            heads=5,            # 멀티 헤드 어텐션에서 사용되는 어텐션 헤드의 개수입니다. 멀티 헤드 어텐션은 모델이 여러 관점에서 정보를 취합할 수 있도록 합니다.
            dim_head=config.seq_len // 2,        # 각 어텐션 헤드의 차원입니다. 어텐션 헤드는 입력 특성을 서로 다른 부분 공간으로 매핑하여 모델이 다양한 특징을 학습할 수 있게 합니다.
            causal=False        # 캐주얼 어텐션 여부를 나타냅니다. 캐주얼 어텐션은 각 위치에서 이전 위치만 참조하도록 하는데, 이것은 주로 시퀀스 데이터에서 다음 값을 예측하는 데 사용됩니다.
        ).to(config.device)

        self.pos_embedding_time = PositionalEmbedding(config.seq_len, config.n_files)

        self.performer_time = Performer(
            dim=config.n_files,
            depth=4,
            heads=124//31,
            dim_head=config.n_files // 2,
            causal=False
        ).to(config.device)

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

        outputs = linear_outputs.permute(0, 2, 1)
        outputs = self.pos_embedding_ticker(outputs)
        outputs = self.performer_ticker(outputs)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.pos_embedding_time(outputs)
        outputs = self.performer_time(outputs)

        outputs = outputs.mean(dim=1)

        return outputs