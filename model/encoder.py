import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(256, hidden_size),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class decoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            # nn.ReLU(),
            # nn.Linear(256, input_size),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    input_size = 207
    hidden_size = 200

    # 创建自编码器模型
    model = encoder(input_size, hidden_size)
    model_de = decoder(hidden_size, input_size)

    # 创建输入数据（假设长度为100）
    input_data = torch.randn(32, 1, 12, input_size)

    # 前向传播
    encoded = model(input_data)
    decoded = model_de(encoded)
    # 输出原始输入和解码后的输出
    print("原始输入: ", input_data)
    print("解码后的输出: ", encoded.shape)
    print("解码后的输出: ", decoded.shape)
    input_data = torch.randn(10, 10)
    print(input_data.shape)
    input_data.resize(30, 30)
    print(input_data.shape)
    