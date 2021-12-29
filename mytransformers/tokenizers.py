import dill as pickle
import fileinput
import io
import numpy as np
import sentencepiece as spm


class SentencePieceTokenizer:
    """class-based wrapper for sentencepiece"""
    
    def __init__(self):
        self.sp = None
    
    
    def fit(self, iterable, vocab_size=8000, control_symbols="[CLS],[SEP],[NEW1],[NEW2],[NEW3]", **kwargs):
        """fit a sentencepiece model on sentence data iterable"""
        try:
            _ = iter(iterable)
        except TypeError as e:
            print('data is not iterable')
            
        _model = io.BytesIO()
        
        spm.SentencePieceTrainer.train(sentence_iterator=iter(iterable), 
                                       model_writer=_model,
                                       vocab_size=vocab_size, 
                                       pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                                       pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
                                       control_symbols=control_symbols.split(","),
                                       **kwargs
                                      )

        self.sp = spm.SentencePieceProcessor(model_proto=_model.getvalue())
        
        return sp.vocab_size()
    
    
    def fit_on_files(self, text_list, vocab_size=8000, control_symbols="[CLS],[SEP],[NEW]", **kwargs):
        """fit a sentencepiece model to one or more sentence-segmented raw text files"""
        if type(text_list) is not list:
            text_list = [text_list]
            
        _model = io.BytesIO()
        _data = fileinput.input(text_list)
        
        spm.SentencePieceTrainer.train(
            sentence_iterator=_data, 
            model_writer=_model,
            vocab_size=vocab_size, 
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
            control_symbols=control_symbols.split(","),
            **kwargs
            )

        self.sp = spm.SentencePieceProcessor(model_proto=_model.getvalue())
        
        return sp.vocab_size()
    
        
    def transform(self, texts, as_array=True, bos=False, eos=False, max_len=None):
        """transform one or more texts into zero-padded arrays"""
        if type(texts) is not list:
            texts = [texts]
        
        # tokenize, optionally add bos, eos
        ids = self.tokenize_as_ids(texts)
        if bos:
            ids = [[2]+x for x in ids]
        if eos:
            ids = [x+[3] for x in ids]
        lens = np.array([len(x) for x in ids])
        
        # get max len. if specified, then truncate
        if max_len is None:
            max_len = np.max(lens)
        elif np.max(lens) > max_len:
            ids = [x[:max_len] for x in ids]
            
        # if specified, output as right-zero-padded array
        if as_array:
            ids = np.array([np.pad(np.array(x), (0, max_len-len(x))) for x in ids])
            
        return ids, lens
    
    
    def inverse_transform(self, ids, as_tokens=False):
        """inverse a list or array of indexed tokens"""
        if type(ids) is np.ndarray:
            ids = list([[int(c) for c in np.trim_zeros(x, "b")] for x in ids])
            
        if not as_tokens:
            return [self.sp.decode(x) for x in ids]
        else:
            return [self.sp.id_to_piece(x) for x in ids]
        
        
    def tokenize_as_string(self, texts):
        """tokenize a (list of) strings, as readable tokens"""
        if self.sp is None:
            raise RuntimeError("no fit tokenizer model, please call fit() first!")
        return self.sp.encode(texts, out_type=str)
    
    
    def tokenize_as_ids(self, texts):
        """tokenize a (list of) strings, as token indices"""
        if self.sp is None:
            raise RuntimeError("no fit tokenizer model, please call fit() first!")
        return self.sp.encode(texts, out_type=int)
    
    
    def export_model(self, file_path):
        """export the inner sentencepiece model for use with sentencepiece library"""
        with open(file_path, "wb") as out:
            out.write(self.sp.serialized_model_proto())
        return True
    
    
    def load_model(self, file_path):
        """load an exported (or otherwise trained) sentencepiece model"""
        self.sp = spm.SentencePieceProcessor(model_file=file_path)
        return True
    
    

