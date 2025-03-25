import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个长为 max_len 的位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 扩展维度以匹配输入 (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 加入位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x

class Mamba_TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(Mamba_TransformerEncoder, self).__init__()
        # 位置编码模块
        self.pos_encoder = PositionalEncoding(embed_dim)
        # 创建多个 EncoderLayerWithCrossAttention 层
        self.layers = nn.ModuleList([
            Mamba_TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # 最后的层规范化
        self.norm = nn.LayerNorm(embed_dim)
        
    
    def forward(self, src, mask=None, cross_mask=None):
        # 输入加上位置编码
        x = self.pos_encoder(src)

        # 依次通过每一层的 encoder layer
        for layer in self.layers:
            x = layer(x, mask=mask, cross_mask=cross_mask)

        # 最后的层规范化
        return self.norm(x)

# 定义 EncoderLayer
class Mamba_TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(Mamba_TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=embed_dim, # Model dimension d_model
                d_state=64,  # SSM state expansion factor, typically 64 or 128
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        # self.ffn2 = nn.Sequential(
        #     nn.Linear(embed_dim, ff_dim),
        #     nn.ReLU(),
        #     nn.Linear(ff_dim, embed_dim),
        # )
    def forward(self, src, mask=None, cross_mask=None):
        # Self-attention
        
        attn_in = src.permute(0, 1, 2)
        self_attn_output, _ = self.self_attn(attn_in, attn_in, attn_in, attn_mask=mask)
        self_attn_output = self_attn_output.permute(0, 1, 2)
        
        mamba_out = self.mamba(src)
        out = self.norm1(self.dropout1(self.ffn1(self_attn_output+mamba_out)))
        # out = self.norm2(stage1 + self.dropout2(self.ffn2(self_attn_output)))
        return out