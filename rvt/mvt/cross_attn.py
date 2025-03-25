import torch
import torch.nn as nn
import math

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

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # 位置编码模块
        # self.pos_encoder = PositionalEncoding(embed_dim)
        # 创建多个 EncoderLayerWithCrossAttention 层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # 最后的层规范化
        # self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src, cross_src, mask=None, cross_mask=None):
        
        if torch.equal(src, cross_src):
            for layer in self.layers:
                src = layer(src, src, mask=mask, cross_mask=cross_mask)
            
        else:
        # 输入加上位置编码
            # src = self.pos_encoder(src)
            # 依次通过每一层的 encoder layer
            for layer in self.layers:
                src = layer(src, cross_src, mask=mask, cross_mask=cross_mask)

        # 最后的层规范化
        return src

# 定义 EncoderLayer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3= nn.Dropout(dropout)

    def forward(self, src, cross_src, mask=None, cross_mask=None):
        if torch.equal(src, cross_src):
        # Self-attention
            src = src.transpose(0, 1)
            self_attn_output, _ = self.self_attn(src, src, src, attn_mask=mask)
            # print(self_attn_output.shape)
            # print(src.shape)
            # exit()
            src = self.norm1(src + self.dropout1(self_attn_output))
            
            ffn_output = self.ffn(src)
            out = self.norm2(src + self.dropout2(ffn_output))
            # src = src.transpose(0, 1)
            
        else:
            src2 = src.transpose(0, 1)
            self_attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=mask)
            # self_attn_output = self_attn_output.transpose(0, 1)
            # print(self_attn_output.shape)
            # exit()
            self_attn_output = self.norm1(src2 + self.dropout1(self_attn_output))

            # Cross-attention
            cross_src2 = cross_src.transpose(0, 1)
   
            cross_attn_output, _ = self.cross_attn(self_attn_output, cross_src2, cross_src2, attn_mask=cross_mask)
            # cross_attn_output = cross_attn_output.transpose(0, 1)
            out = self.norm2(src2 + self.dropout2(cross_attn_output))

            # Feedforward network
            ffn_output = self.ffn(out)
            out = self.norm3(out + self.dropout3(ffn_output))
            
        return out.permute(1, 0, 2)