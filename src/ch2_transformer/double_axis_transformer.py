import torch
from torch import nn

class StockTransformerModel(nn.Module):
    def __init__(self, input_size, output_size, nhead):
        super(StockTransformerModel, self).__init__()

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



######################################################################
