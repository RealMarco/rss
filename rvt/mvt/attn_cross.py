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

    def forward(self, src, cross_src):
        
        if torch.equal(src, cross_src):
            for layer in self.layers:
                src = layer(src, src)
            
        else:
        # 输入加上位置编码
            # src = self.pos_encoder(src)
            # 依次通过每一层的 encoder layer
            for layer in self.layers:
                src = layer(src, cross_src)

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

    def forward(self, src, cross_src):
        src1 = src.transpose(0, 1)
        
        if torch.equal(src, cross_src):
            self_attn_output, _ = self.self_attn(src1, src1, src1)
            src1 = self.norm1(src1 + self.dropout1(self_attn_output))
            ffn_output = self.ffn(src1)
            out = self.norm2(src1 + self.dropout2(ffn_output))
            # src = src.transpose(0, 1)
            
        else:
            self_attn_output, _ = self.self_attn(src1, src1, src1)
            self_attn_output = self.norm1(src1 + self.dropout1(self_attn_output))
            # if torch.isnan(self_attn_output).any():
            #     print("self_attn_output____张量中存在 NaN 值")
            #     print("src", self_attn_output.transpose(0, 1)[0])
            #     exit()

            # Cross-attention
            cross_src1 = cross_src.transpose(0, 1)
            # if torch.isnan(cross_src1).any():
            #     print("cross_src1____张量中存在 NaN 值")
            #     print("src", cross_src1.transpose(0, 1)[0])
            #     exit()
        
            cross_attn_output, _ = self.cross_attn(self_attn_output, cross_src1, cross_src1)
            # if torch.isnan(cross_attn_output).any():
            #     print("cross_attn_output____张量中存在 NaN 值")
            #     exit()
            # cross_attn_output = cross_attn_output.transpose(0, 1)
            out = self.norm2(src1 + self.dropout2(cross_attn_output))
            # if torch.isnan(cross_attn_output).any():
            #     print("out____张量中存在 NaN 值")
            #     exit()
            # Feedforward network
            ffn_output = self.ffn(out)
            # if torch.isnan(ffn_output).any():
            #     print("ffn____张量中存在 NaN 值")
            #     exit()
            out = self.norm3(out + self.dropout3(ffn_output))
            # if torch.isnan(out).any():
            #     print("outreturn____张量中存在 NaN 值")
            #     exit()
            
        return out.transpose(0, 1)
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 使用 batch_first=True 以支持 (batch_size, sequence_length, dim) 的输入
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.encoder_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # 层归一化和 Dropout
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        tgt2 = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm1(tgt)
        
        # Cross-attention (encoder-decoder attention)
        tgt2 = self.encoder_attention(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # 定义解码器的层和位置编码
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 添加位置编码
        tgt = self.positional_encoding(tgt)
        
        # 逐层传递
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        return self.norm(tgt)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建 (max_len, d_model) 的位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加批次维度以便广播
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 根据输入 x 的大小自动裁剪位置编码
        return x + self.pe[:, :x.size(1), :]