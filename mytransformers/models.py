import inspect
import numpy as np
import torch

from mytransformers.modules import LearnedPositionalEncoding
from mytransformers.modules import SinusoidalPositionalEncoding
from mytransformers.layers import TransformerEncoderLayer
from mytransformers.layers import TransformerDecoderLayer


class TransformerModel(torch.nn.Module):
    """basic seq2seq transformer from `Attention is All You Need`
    
    options/modifications:
    - can embed vocabulary at lower dimension and project, like ALBERT
    - can share vocabulary and embeddings in encoder and decoder
    """
    
    def __init__(self,
                 src_vocab_sz: int,
                 tgt_vocab_sz: int,
                 enc_layers: int,
                 dec_layers: int,
                 seq_len: int,
                 d_vocab: int,
                 d_in: int, 
                 d_attn: int, 
                 d_ffnn: int, 
                 attn_heads: int, 
                 attn_dropout: float, 
                 ffnn_dropout: float,
                 pos_encoding: str="sinusoidal",
                 shared_vocab: bool=False,
                 attn_mask_val: float=-10e8, 
                 attn_q_bias: bool=False, 
                 attn_kv_bias: bool=False, 
                 attn_out_bias: bool=False, 
                 ffnn_activation: str="relu", 
                 pre_ln: bool=False
                ):
        # save initial configuration
        frame = inspect.currentframe()
        keys, _, _, values = inspect.getargvalues(frame)
        self.config = {}
        for key in keys:
            if key != 'self':
                self.config[key] = values[key]
        super().__init__()
        
        # positional encoding
        if pos_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding()
        elif pos_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding()
        else:
            raise ValueError("'pos_encoding' must be in ('sinusoidal', 'learned')!")
        
        # vocabulary embedding layer(s)
        self.shared_vocab = shared_vocab
        self.seq_len = seq_len
        if d_vocab != d_in:
            self.proj_to = torch.nn.Linear(d_vocab, d_in)
            self.proj_from = torch.nn.Linear(d_in, d_vocab)
            if not self.shared_vocab:
                self.tgt_proj_to = torch.nn.Linear(d_vocab, d_in)
        else:
            self.proj_to = torch.nn.Identity()
            self.proj_from = torch.nn.Identity()
            if not self.shared_vocab:
                self.tgt_proj_to = torch.nn.Identity()

        self.embedding = torch.nn.Embedding(src_vocab_sz, d_vocab)
        if not self.shared_vocab:
            self.tgt_embedding = torch.nn.Embedding(tgt_vocab_sz, d_vocab)
            
        # encoder layers
        _encoder_layers = []
        if enc_layers < 1:
            raise ValueError("'enc_layers' must be > 1!")
        for lyr_idx in range(enc_layers):
            l = TransformerEncoderLayer(
                 seq_len=seq_len,
                 d_in=d_in, 
                 d_attn=d_attn, 
                 d_ffnn=d_ffnn, 
                 attn_heads=attn_heads, 
                 attn_dropout=attn_dropout, 
                 ffnn_dropout=ffnn_dropout,
                 attn_mask_val=attn_mask_val, 
                 attn_q_bias=attn_q_bias, 
                 attn_kv_bias=attn_kv_bias, 
                 attn_out_bias=attn_out_bias, 
                 ffnn_activation=ffnn_activation, 
                 pre_ln=pre_ln
            )
            _encoder_layers.append(l)
        self.encoder_layers = torch.nn.ModuleList(_encoder_layers)
        
        # encoder layers
        _decoder_layers = []
        if enc_layers < 1:
            raise ValueError("'dec_layers' must be > 1!")
        for lyr_idx in range(dec_layers):
            l = TransformerDecoderLayer(
                 seq_len=seq_len,
                 d_in=d_in, 
                 d_attn=d_attn, 
                 d_ffnn=d_ffnn, 
                 attn_heads=attn_heads, 
                 attn_dropout=attn_dropout, 
                 ffnn_dropout=ffnn_dropout,
                 attn_mask_val=attn_mask_val, 
                 attn_q_bias=attn_q_bias, 
                 attn_kv_bias=attn_kv_bias, 
                 attn_out_bias=attn_out_bias, 
                 ffnn_activation=ffnn_activation, 
                 pre_ln=pre_ln
            )
            _decoder_layers.append(l)
        self.decoder_layers = torch.nn.ModuleList(_decoder_layers)
        
    
    def get_config(self):
        
        return self.config
    
    
    def forward(self, x, y_in, x_lens=None, y_lens=None):

        # encode inputs
        x = self.embedding(x)
        x = self.proj_to(x)
        x = self.positional_encoding(x)
        for enc_lyr in self.encoder_layers:
            x = enc_lyr(x, seq_lens=x_lens)
        
        # decoder with shifted inputs
        if self.shared_vocab:
            _emb = self.embedding
            _prj = self.proj_to
        else:
            _emb = self.tgt_embedding
            _prj = self.tgt_proj_to
        y = _emb(y_in)
        y = _prj(y)
        y = self.positional_encoding(y)
        for dec_lyr in self.decoder_layers:
            x = dec_lyr(x, y, mem_lens=x_lens, seq_lens=y_lens)
        
        # vocabulary projection with weight tying
        y = self.proj_from(y)
        if self.shared_vocab:
            w_T = self.embedding.weight.t()
        else:
            w_T = self.tgt_embedding.weight.t()
        y = torch.matmul(y, w_T)
        
        return y