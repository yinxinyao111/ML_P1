import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # result holder (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # pos (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = float).unsqueeze(1) 
        # div_term (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # apply sine to even indices (seq_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices (seq_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)
        # add batch dimension to pe (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        # resigter pe as buffer
        self.register_buffer("pe", pe)
    def forward(self, x):
        # (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6):
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # (d_model)
        self.bias = nn.Parameter(torch.zeros(features)) # (d_model)
    def forward(self, x):
        # (batch, seq_len, 1)
        mean = x.mean(dim = -1, keepdim = True)
        # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model currently is not divisible by h"
    
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
        # apply mask
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        # softmax in between
        attention_scores = attention_scores.softmax(dim = -1)
        # apply dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0, -1, self.h * self.d_k])
        
        # multiply by w_o (batch, seq_len, d_model)
        return self.w_o(x)
    
# basic units are defined above, now compile them to encoders and decoders
# ----------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block)