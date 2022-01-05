import torch
import numpy as np

from mytransformers.modules import MultiHeadAttention
from mytransformers.modules import PositionWiseFeedForward
from mytransformers.modules import SinusoidalPositionalEncoding


class TransformerEncoderLayer(torch.nn.Module):
    """transformer encoder layer
    
    optionally, this can be configured as "pre-LN" Transformer
    which moves the LayerNorm up in the residual sub-blocks
    """
    
    def __init__(self, 
                 seq_len: int,
                 d_in: int, 
                 d_attn: int,
                 d_ffnn: int, 
                 attn_heads: int, 
                 attn_dropout: float, 
                 ffnn_dropout: float,
                 attn_mask_val: float=-1e08, 
                 attn_q_bias: bool=False, 
                 attn_kv_bias: bool=False, 
                 attn_out_bias: bool=False, 
                 ffnn_activation: str="relu", 
                 pre_ln: bool=False):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_in = d_in
        self.pre_ln = pre_ln
        
        self.mha = MultiHeadAttention(seq_len, d_in, d_attn, attn_heads, 
                                      dropout=attn_dropout, 
                                      causal=False,
                                      mask_val=attn_mask_val, 
                                      q_bias=attn_q_bias,
                                      kv_bias=attn_kv_bias,
                                      out_bias=attn_out_bias)
        
        self.ffnn = PositionWiseFeedForward(d_in, d_ffnn, 
                                            activation=ffnn_activation,
                                            dropout=ffnn_dropout)
        
        self.layer_norm_1 = torch.nn.LayerNorm(d_in)
        self.layer_norm_2 = torch.nn.LayerNorm(d_in)
        
        
    def forward(self, x, seq_lens=None, mask_output=True):
        
        assert x.shape[1] == self.seq_len
        assert x.shape[2] == self.d_in
            
        # multi-head attention sub-block
        if self.pre_ln:
            r = x
            x = self.layer_norm_1(x)
            _, x = self.mha(x, x, x, q_lens=seq_lens, kv_lens=seq_lens)
            x += r
        else:
            r = x
            _, x = self.mha(x, x, x, q_lens=seq_lens, kv_lens=seq_lens)
            x += r
            x = self.layer_norm_1(x)
            
        # positionwise FFNN sub-block
        if self.pre_ln:
            r = x
            x = self.layer_norm_2(x)
            x = self.ffnn(x)
            x += r
        else:
            r = x
            x = self.ffnn(x)
            x += r
            x = self.layer_norm_2(x)
            
        # apply 0 mask based on max sequence lengths 
        if mask_output and seq_lens is not None:
            x_pad = torch.arange(x.shape[1])[None, :].expand(x.shape[0], -1).to(x.device) >= seq_lens[:, None]
            x.masked_fill_(x_pad.unsqueeze(-1), 0.0)

        return x
    
    
class TransformerDecoderLayer(torch.nn.Module):
    """transformer decoder layer
    
    optionally, this can be configured as "pre-LN" Transformer
    which moves the LayerNorm up in the residual sub-blocks
    """
    
    def __init__(self, 
                 seq_len: int,
                 d_in: int, 
                 d_attn: int, 
                 d_ffnn: int, 
                 attn_heads: int, 
                 attn_dropout: float, 
                 ffnn_dropout: float,
                 attn_mask_val: float=-1e08, 
                 attn_q_bias: bool=False, 
                 attn_kv_bias: bool=False, 
                 attn_out_bias: bool=False, 
                 ffnn_activation: str="relu", 
                 pre_ln: bool=False):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_in = d_in
        self.pre_ln = pre_ln
        
        # masked (causal) self-attention
        self.masked_mha = MultiHeadAttention(seq_len, d_in, d_attn, attn_heads, 
                                             dropout=attn_dropout, 
                                             causal=True,
                                             mask_val=attn_mask_val, 
                                             q_bias=attn_q_bias,
                                             kv_bias=attn_kv_bias,
                                             out_bias=attn_out_bias)
        
        # encoder (memory) attention
        self.mha = MultiHeadAttention(seq_len, d_in, d_attn, attn_heads, 
                                      dropout=attn_dropout, 
                                      causal=False,
                                      mask_val=attn_mask_val, 
                                      q_bias=attn_q_bias,
                                      kv_bias=attn_kv_bias,
                                      out_bias=attn_out_bias)
        
        self.ffnn = PositionWiseFeedForward(d_in, d_ffnn, 
                                            activation=ffnn_activation,
                                            dropout=ffnn_dropout)
        
        self.layer_norm_1 = torch.nn.LayerNorm(d_in)
        self.layer_norm_2 = torch.nn.LayerNorm(d_in)
        self.layer_norm_3 = torch.nn.LayerNorm(d_in)
        
        
    def forward(self, mem, x, mem_lens=None, seq_lens=None, mask_output=True):
        
        assert x.shape[1] == self.seq_len
        assert x.shape[2] == self.d_in
        assert mem.shape[1] == self.seq_len
        assert mem.shape[2] == self.d_in
            
        # causal self-attention sub-block
        if self.pre_ln:
            r = x
            x = self.layer_norm_1(x)
            es, x = self.masked_mha(x, x, x, q_lens=seq_lens, kv_lens=seq_lens)
            x += r
        else:
            r = x
            es, x = self.masked_mha(x, x, x, q_lens=seq_lens, kv_lens=seq_lens)
            x += r
            x = self.layer_norm_1(x)
            
        # memory attention
        if self.pre_ln:
            r = x
            x = self.layer_norm_2(x)
            ds, x = self.mha(x, mem, mem, q_lens=seq_lens, kv_lens=mem_lens)
            x += r
        else:
            r = x
            ds, x = self.mha(x, mem, mem, q_lens=seq_lens, kv_lens=mem_lens)
            x += r
            x = self.layer_norm_2(x)
            
        # positionwise FFNN sub-block
        if self.pre_ln:
            r = x
            x = self.layer_norm_3(x)
            x = self.ffnn(x)
            x += r
        else:
            r = x
            x = self.ffnn(x)
            x += r
            x = self.layer_norm_3(x)
            
        # apply 0 mask based on max sequence lengths 
        if mask_output and seq_lens is not None:
            x_pad = torch.arange(x.shape[1])[None, :].expand(x.shape[0], -1).to(x.device) >= seq_lens[:, None]
            x.masked_fill_(x_pad.unsqueeze(-1), 0.0)
            
        return x
