import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFeatureEnhancer0(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super(TransformerFeatureEnhancer, self).__init__()
        # Transformer 的编码器结构
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 特征维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=dim_feedforward,  # 前馈网络维度
            dropout=dropout  # dropout 比率
        )
        # Transformer 编码器，由多个编码器层组成
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # x 是形状 (B, 128, T)，我们需要把它变为 (T, B, 128) 的形状，因为 PyTorch 的 Transformer 要求 (序列长度, batch_size, 特征维度)
        x = x.permute(2, 0, 1)  # (T, B, 128)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # (B, 128, T)

        return x


class TransformerFeatureEnhancer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1, max_len=5000):
        super(TransformerFeatureEnhancer, self).__init__()
        self.d_model = d_model

        # 创建一个固定位置编码（或学习型位置编码）
        self.position_offset = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len, 1)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # x 是形状 (B, 128, T)，需要变为 (T, B, 128) 形式
        x = x.permute(2, 0, 1)  # (T, B, 128)

        # 获取每个位置的可学习位置编码，并扩展成 (T, 1, d_model) 的形状
        seq_len = x.size(0)
        position_enc = self.position_offset[:seq_len, :].to(x.device)  # (T, 1)
        position_enc = position_enc.unsqueeze(1)  # (T, 1, 128) 扩展成 (T, 1, d_model)

        # 通过广播机制将位置编码添加到 x 上
        x = x + position_enc  # (T, B, 128)

        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # (B, 128, T)

        return x


