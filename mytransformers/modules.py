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
            self.register_buffer("mask", torch.zeros((1, self.seq_len, self.seq_len)).bool())
        
    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        device = self.dummy
        b, l, d = q.shape
        kb, kl, ki = k.shape
        vb, vl, vi = k.shape
        assert kb == vb and kl == vl and ki == vi
        assert b == kb
        assert l == kl
        assert d == ki == self.d_in
        
        # linear projection, split into heads, and transpose to (b * h, l, d_attn)
        _q = self.dropout(self.proj_q(q)).transpose(1, 2).contiguous().view(b * self.heads, l, self.d_attn)
        _k = self.dropout(self.proj_k(k)).transpose(1, 2).contiguous().view(b * self.heads, l, self.d_attn)
        _v = self.dropout(self.proj_v(v)).transpose(1, 2).contiguous().view(b * self.heads, l, self.d_attn)
        
        # scaled dot product: softmax( Q @ K.T / sqrt(d_k) ) * V >> (b * h, l, l)
        _qk = torch.bmm(_q, _k.transpose(1, 2))/np.sqrt(self.d_attn)
        
        # rehape masks to b* h
        q_mask = q_mask.unsqueeze(1).expand(-1, self.heads, -1).reshape(b * self.heads, -1)
        kv_mask = kv_mask.unsqueeze(1).expand(-1, self.heads, -1).reshape(b * self.heads, -1).unsqueeze(1)
        
        # apply padding and sequence masks
        _qk.masked_fill_(self.mask, self.mask_val)
        if q_mask is not None:
            _qk.masked_fill_(q_mask, self.mask_val)
        if kv_mask is not None:
            _qk.masked_fill_(kv_mask, self.mask_val)
        scores = torch.nn.functional.softmax(_qk, dim=-1)  # across each row index 2
        _attn = torch.bmm(scores, _v).view(b, self.heads, l, self.d_attn)  # (b * h, l, d) >> (b, h, l, d)
        
        # transpose to (b, l, h, d_attn), reshape to (b, l, h*d_attn), project to (b, l, d_in)
        _attn = _attn.transpose(1, 2).contiguous().view(b, l, self.heads * self.d_attn)
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

    def forward(self, embeddings):
        # create positional encoding
        embeddings *= np.sqrt(embeddings.shape[-1])  # from "Attention..." section 3.4
        embeddings += self.learned_embeddings.unsqueeze(0).repeat(embeddings.shape[0], 1, 1)
        
        return embeddings
    

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, d_in):
        super().__init__()
        # create positional encoding
        _pos_encoding = torch.arange(0, seq_len).unsqueeze(1).repeat(1, d_in).float()  # index each step
        denominator = 1. / (10000 ** (torch.arange(0, d_in, 2).float() / d_in))        # denominator of inner term
        _pos_encoding[:, 0::2] = torch.sin(_pos_encoding[:, 0::2] * denominator)  # apply sine to every other
        _pos_encoding[:, 1::2] = torch.cos(_pos_encoding[:, 1::2] * denominator)  # apply cosine to every other
        self.register_buffer("pos_encoding", _pos_encoding)
        
    def forward(self, embeddings):
        embeddings *= np.sqrt(embeddings.shape[-1])  # from "Attention..." section 3.4
        embeddings += self.pos_encoding.unsqueeze(0).repeat(embeddings.shape[0], 1, 1)
        
        return embeddings
    