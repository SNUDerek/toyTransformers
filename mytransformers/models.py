import inspect
import numpy as np
import torch
import tqdm

from mytransformers.functions import get_mask_from_lengths
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
                 dropout: float=0.1,
                 attn_dropout: float=0.0, 
                 ffnn_dropout: float=0.0,
                 pos_encoding: str="sinusoidal",
                 shared_vocab: bool=False,
                 attn_mask_val: float=-1e08, 
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
        self.src_vocab_sz = src_vocab_sz
        self.tgt_vocab_sz = tgt_vocab_sz
        self.shared_vocab = shared_vocab
        self.seq_len = seq_len
        self.d_in = d_in
        
        # positional encoding
        if pos_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(self.seq_len, self.d_in)
        elif pos_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(self.seq_len, self.d_in)
        else:
            raise ValueError("'pos_encoding' must be in ('sinusoidal', 'learned')!")
        
        # vocabulary embedding layer(s)
        self.embedding = torch.nn.Embedding(src_vocab_sz, d_vocab)
        if not self.shared_vocab:
            self.tgt_embedding = torch.nn.Embedding(tgt_vocab_sz, d_vocab)
            
        if d_vocab != d_in:
            self.proj_to = torch.nn.Linear(d_vocab, d_in, bias=False)
            self.proj_from = torch.nn.Linear(d_in, d_vocab, bias=False)
            if not self.shared_vocab:
                self.tgt_proj_to = torch.nn.Linear(d_vocab, d_in, bias=False)
        else:
            self.proj_to = torch.nn.Identity()
            self.proj_from = torch.nn.Identity()
            if not self.shared_vocab:
                self.tgt_proj_to = torch.nn.Identity()
                
        self.dropout = torch.nn.Dropout(dropout)

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
        x = self.dropout(x)
        for enc_lyr in self.encoder_layers:
            x = enc_lyr(x, seq_lens=x_lens)
            x = self.dropout(x)
        # zero-mask the padded indices
        if x_lens is not None:
            x_pad = get_mask_from_lengths(x_lens, self.seq_len, x.device)
            x = torch.masked_fill(x, x_pad.unsqueeze(-1), 0.0)
        
        # decoder with shifted inputs
        if self.shared_vocab:
            _emb = self.embedding
            _prj = self.proj_to
            _siz = self.src_vocab_sz
        else:
            _emb = self.tgt_embedding
            _prj = self.tgt_proj_to
            _siz = self.tgt_vocab_sz
        y = _emb(y_in)
        y = _prj(y)
        y = self.positional_encoding(y)
        y = self.dropout(y)
        for dec_lyr in self.decoder_layers:
            y = dec_lyr(x, y, mem_lens=x_lens, seq_lens=y_lens)
            y = self.dropout(y)
        
        # vocabulary projection with weight tying
        y = self.proj_from(y)
        if self.shared_vocab:
            w = self.embedding.weight
        else:
            w = self.tgt_embedding.weight
        y = torch.matmul(y, w.t())
        
        return y
    
    
    def infer_one_greedy(self, x, x_lens, bos=2, eos=3, verbose=False):
        
        # dummy input
        y_in = torch.zeros_like(x).long().to(x.device)
        y_in[0][0] = bos
        y_lens = torch.from_numpy(np.array([1])).long().to(x.device)
        
        # encode inputs
        x = self.embedding(x)
        x = self.proj_to(x)
        x = self.positional_encoding(x)
        for enc_lyr in self.encoder_layers:
            x = enc_lyr(x, seq_lens=x_lens)
        # zero-mask the padded indices
        if x_lens is not None:
            x_pad = get_mask_from_lengths(x_lens, self.seq_len, x.device)
            x = x.masked_fill(x_pad.unsqueeze(-1), 0.0)

        rng = tqdm.trange(self.seq_len-1) if verbose else range(self.seq_len-1)
            
        for i in rng:

            # decoder with shifted inputs
            if self.shared_vocab:
                _emb = self.embedding
                _prj = self.proj_to
                _siz = self.src_vocab_sz
            else:
                _emb = self.tgt_embedding
                _prj = self.tgt_proj_to
                _siz = self.tgt_vocab_sz
            y = _emb(y_in)
            y = _prj(y)
            y = self.positional_encoding(y)
            for dec_lyr in self.decoder_layers:
                y = dec_lyr(x, y, mem_lens=x_lens, seq_lens=y_lens)

            # vocabulary projection with weight tying
            y = self.proj_from(y)
            if self.shared_vocab:
                w = self.embedding.weight
            else:
                w = self.tgt_embedding.weight
            y = torch.matmul(y, w.t())

            # get next predicted token by softmax
            y_p = torch.nn.functional.softmax(y, dim=-1)
            next_token = torch.argmax(y_p[0][i])
            y_in[0][i+1] = next_token
            y_lens[0] += 1
            if next_token == eos:
                break

        return y_in[0].detach().cpu().numpy().tolist()