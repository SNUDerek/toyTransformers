import numpy as np
import torch
from functools import reduce
from operator import iadd


class MultiHeadAttention(torch.nn.Module):
    """multi-head attention"""
    
    def __init__(self, seq_len: int, d_in: int, d_attn: int, heads: int,
                 dropout: float=0.0, causal: bool=False, mask_val=-10e8,
                 q_bias: bool=False, kv_bias: float=False, out_bias: float=False):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_in = d_in
        self.d_attn = d_attn
        self.heads = heads
        self.mask_val = mask_val
        self.dummy = torch.nn.Parameter(torch.empty(0))
        
        self.proj_q = torch.nn.Linear(self.d_in, self.d_attn * self.heads, bias=q_bias)
        self.proj_k = torch.nn.Linear(self.d_in, self.d_attn * self.heads, bias=kv_bias)
        self.proj_v = torch.nn.Linear(self.d_in, self.d_attn * self.heads, bias=kv_bias)
        self.proj_o = torch.nn.Linear(self.heads * self.d_attn, self.d_in, bias=out_bias)
        self.dropout = torch.nn.Dropout(dropout)
        
        if causal:
            self.register_buffer("mask", torch.triu(torch.ones((1, self.seq_len, self.seq_len)), diagonal=1).bool())
        else:
            self.register_buffer("mask", torch.ones((1, self.seq_len, self.seq_len)).bool())
        
    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        device = self.dummy
        b, l, i = q.shape
        kb, kl, ki = k.shape
        assert b == kb
        assert l == kl
        assert i == ki == self.d_in
        
        # linear projection, split into heads, and transpose to (b, h, l, d_attn)
        _q = self.proj_q(q).reshape((b, l, self.heads, self.d_attn)).transpose(1, 2)
        _k = self.proj_k(k).reshape((b, l, self.heads, self.d_attn)).transpose(1, 2)
        _v = self.proj_v(v).reshape((b, l, self.heads, self.d_attn)).transpose(1, 2)
        
        # scaled dot product: softmax( Q @ K.T / sqrt(d_k) ) * V
        _qk = torch.matmul(_q, _k.transpose(-2, -1))/np.sqrt(self.d_attn)
        
        # apply padding and sequence masks
        _qk = _qk.masked_fill(self.mask, self.mask_val)
        _qk = self.dropout(_qk)
        if q_mask is not None:
            _qk = _qk.masked_fill(q_mask.unsqueeze(1).unsqueeze(-1), self.mask_val)
        if kv_mask is not None:
            _qk = _qk.masked_fill(kv_mask.unsqueeze(1).unsqueeze(1), self.mask_val)
        scores = torch.nn.functional.softmax(_qk, dim=-1)
        _attn = torch.matmul(scores, _v)
        
        # transpose to (b, l, h, d_attn), reshape to (b, l, h*d_attn), project to (b, l, d_in)
        _attn = torch.reshape(_attn.transpose(1, 2), (b, l, self.heads * self.d_attn))
        output = self.proj_o(_attn)
        
        return scores, output
    

class PositionWiseFeedForward(torch.nn.Module):
    """position-wise FFNN"""
    
    def __init__(self, d_in: int, d_ffn: int, activation: str="relu", dropout: float=0.0):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_ffn)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ffn, d_in)
        if activation.lower() == "relu":
            self.activation = torch.nn.ReLU()
        elif activation.lower() == "selu":
            self.activation = torch.nn.SELU()
        elif activation.lower() == "gelu":
            self.activation = torch.nn.GELU()
        
    def forward(self, x):
        x = self.linear2(self.activation(self.dropout(self.linear1(x))))
        return x


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, d_in):
        super().__init__()
        # learned embedding matrix for one sample
        self.learned_embeddings = torch.nn.Parameter(torch.rand(seq_len, d_in).float())
        torch.nn.init.xavier_normal_(self.learned_embeddings)

    def forward(self, embeddings):
        b, l, d = embeddings.shape
        device = embeddings.device
        
        # create positional encoding
        embeddings *= np.sqrt(embeddings.shape[-1])  # from "Attention..." 3.4
        embeddings += self.learned_embeddings.unsqueeze(0).repeat(b, 1, 1)
        
        return embeddings
    

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, embeddings):
        b, l, d = embeddings.shape
        device = embeddings.device
        
        # create positional encoding
        pos_encoding = torch.arange(0, l).unsqueeze(1).repeat(1, d).float().to(device)  # index each step
        denominator = 1. / (10000 ** (torch.arange(0, d, 2).float() / d)).to(device)    # denominator of inner term
        pos_encoding[:, 0::2] = torch.sin(pos_encoding[:, 0::2] * denominator)  # apply sine to every other
        pos_encoding[:, 1::2] = torch.cos(pos_encoding[:, 1::2] * denominator)  # apply cosine to every other
        embeddings *= np.sqrt(embeddings.shape[-1])  # from "Attention..." 3.4
        embeddings += pos_encoding.unsqueeze(0).repeat(b, 1, 1)
        
        return embeddings
    