import torch
import torch.nn as nn


class STGCAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STGCAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class STGCAutodecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STGCAutodecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x
    

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
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
            nn.ReLU(),
            # nn.Linear(256, input_size),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # 创建自编码器模型
    input_dim = 1  # 输入维度
    hidden_dim = 32  # 隐藏层维度
    output_dim = input_dim + 1 # 输出维度与输入维度相同
    autoencoder = STGCAutoencoder(input_dim, hidden_dim, output_dim)

    # 随机生成输入数据
    batch_size = 16
    seq_length = 12
    num_vertices = 207
    input_data = torch.randn(batch_size, input_dim, seq_length, num_vertices)
    print(input_data.shape)
    # 使用自编码器进行前向传播
    reconstructed_data = autoencoder(input_data)

    # 打印重构数据的形状
    print("Reconstructed data shape:", reconstructed_data.shape)
    print("Reconstructed data shape:", reconstructed_data.shape)
    exit(0)
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
    