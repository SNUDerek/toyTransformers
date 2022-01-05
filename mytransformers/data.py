import fileinput
import io
import numpy as np
import sentencepiece as spm
import torch


def pad_to_seq_len(batch, max_seq_len):
    """batch variable-length sequences and right-zero-pad"""
    x_list, y_list, x_len_list, y_len_list = list(zip(*batch))
    x_len_list = [min(x, max_seq_len) for x in x_len_list]
    y_len_list = [min(y, max_seq_len) for y in y_len_list]
    x_array = np.array([np.pad(np.array(x[:max_seq_len]), (0, max_seq_len-len(x))) for x in x_list])
    y_array = np.array([np.pad(np.array(y[:max_seq_len+1]), (0, max_seq_len-len(y)+1)) for y in y_list])
    x_tensor = torch.from_numpy(x_array).long()
    y_in_tensor = torch.from_numpy(y_array[:,:-1]).long()
    y_out_tensor = torch.from_numpy(y_array[:,1:]).long()
    x_lens_tensor = torch.from_numpy(np.array(x_len_list)).long()
    y_lens_tensor = torch.from_numpy(np.array(y_len_list)).long()
    return x_tensor, y_in_tensor, y_out_tensor, x_lens_tensor, y_lens_tensor


class SentencePieceTokenizer:
    """class-based wrapper for sentencepiece"""
    
    def __init__(self):
        self.sp = None
    
    
    def fit(self, iterable, vocab_size=8000, control_symbols="[CLS],[SEP],[MASK],[NEW1],[NEW2]", **kwargs):
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
        
        return self.sp.vocab_size()
    
    
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
        
        return self.sp.vocab_size()
    
        
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
    
    
    def get_vocabulary(self):
        """get token list of (id, token) tuples"""
        return [(id, self.sp.IdToPiece(id)) for id in range(self.sp.GetPieceSize())]
    
    
    def export_model(self, file_path):
        """export the inner sentencepiece model for use with sentencepiece library"""
        with open(file_path, "wb") as out:
            out.write(self.sp.serialized_model_proto())
        return True
    
    
    def export_vocab(self, file_path):
        """export the inner sentencepiece model for use with sentencepiece library"""
        vocab = self.get_vocabulary()
        with open(file_path, "w") as out:
            for i, v in enumerate(vocab):
                if i > 0:
                    out.write("\n")
                out.write("{}\t{}".format(v[0], v[1]))
        return True
    
    
    def load_model(self, file_path):
        """load an exported (or otherwise trained) sentencepiece model"""
        self.sp = spm.SentencePieceProcessor(model_file=file_path)
        return True
    

class SimpleTranslationDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, source_file, target_file, src_tokenizer=None, tgt_tokenizer=None, share_tokenizer=False, sep="\t", verbose=True, **kwargs):
        """simple dataset for translation"""
        self.share_tokenizer = share_tokenizer
        self.source = open(source_file).read().split("\n")
        self.target = open(target_file).read().split("\n")
        if len(self.source[-1]) < 1:
            self.source = self.source[:-1]
        if len(self.target[-1]) < 1:
            self.target = self.target[:-1]
        if len(self.source) != len(self.target):
            raise ValueError("source file len '{}' != target file len '{}'!".format(len(self.source), len(self.target)))
        if src_tokenizer is None:
            if verbose: print("fitting tokenizer...")
            if self.share_tokenizer:
                self.src_tokenizer = SentencePieceTokenizer()
                _ = self.src_tokenizer.fit(self.source + self.target, **kwargs)
                self.tgt_tokenizer = self.src_tokenizer
            else:
                if verbose: print("fitting source tokenizer...")
                self.src_tokenizer = SentencePieceTokenizer()
                _ = self.src_tokenizer.fit(self.source, **kwargs)
                if tgt_tokenizer is None:
                    if verbose: print("fitting target tokenizer...")
                    self.tgt_tokenizer = SentencePieceTokenizer()
                    _ = self.tgt_tokenizer.fit(self.target, **kwargs)
        else:
            self.src_tokenizer = src_tokenizer
            if tgt_tokenizer is None:
                if verbose: print("fitting target tokenizer...")
                self.tgt_tokenizer = SentencePieceTokenizer()
                _ = self.tgt_tokenizer.fit(self.target, **kwargs)
            else:
                self.tgt_tokenizer = tgt_tokenizer
                
                
    def preview(self):
        for index in [0, len(self.source)//2, len(self.source)-1]:
            print("index {}".format(index))
            print("\tsource input : {}".format(self.source[index]))
            print("\ttarget input : {}".format(self.target[index]))
            print("\tsource tokens: {}".format(" ".join(self.src_tokenizer.tokenize_as_string(self.source[index]))))
            print("\ttarget tokens: {}".format(" ".join(self.tgt_tokenizer.tokenize_as_string(self.target[index]))))
            print("---------------------------------------------------------------------------------------------------------")
                    
                    
    def get_tokenizers(self):
                    
        return self.src_tokenizer, self.tgt_tokenizer
                    
        
    def __len__(self):
        return len(self.source)
    

    def __getitem__(self, idx):

        if self.share_tokenizer:
            x, x_len = self.src_tokenizer.transform(self.source[idx], as_array=False, bos=True, eos=True)
            y, y_len = self.src_tokenizer.transform(self.target[idx], as_array=False, bos=True, eos=True)
        else:
            x, x_len = self.src_tokenizer.transform(self.source[idx], as_array=False, bos=True, eos=True)
            y, y_len = self.tgt_tokenizer.transform(self.target[idx], as_array=False, bos=True, eos=True)

        return x[0], y[0], x_len[0], y_len[0]

    
class SimpleCSVTranslationDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_filepath, src_tokenizer=None, tgt_tokenizer=None, share_tokenizer=False, sep="\t", verbose=True, **kwargs):
        """simple dataset for translation"""
        self.share_tokenizer = share_tokenizer
        self.source = []
        self.target = []
        with open(csv_filepath) as f:
            lines = f.read().split("\n")
        for line in lines:
            s, t = line.split(sep)
            if len(s) > 0 and len(t) > 0:
                self.source.append(s)
                self.target.append(t)
        if src_tokenizer is None:
            if verbose: print("fitting tokenizer...")
            if self.share_tokenizer:
                self.src_tokenizer = SentencePieceTokenizer()
                _ = self.src_tokenizer.fit(self.source + self.target, **kwargs)
            else:
                if verbose: print("fitting source tokenizer...")
                self.src_tokenizer = SentencePieceTokenizer()
                _ = self.src_tokenizer.fit(self.source, **kwargs)
                if tgt_tokenizer is None:
                    if verbose: print("fitting target tokenizer...")
                    self.tgt_tokenizer = SentencePieceTokenizer()
                    _ = self.tgt_tokenizer.fit(self.target, **kwargs)
        else:
            self.src_tokenizer = src_tokenizer
            if tgt_tokenizer is None:
                if verbose: print("fitting target tokenizer...")
                self.tgt_tokenizer = SentencePieceTokenizer()
                _ = self.tgt_tokenizer.fit(self.target, **kwargs)
            else:
                self.tgt_tokenizer = tgt_tokenizer
                    
                    
    def get_tokenizers(self):
                    
        return self.src_tokenizer, self.tgt_tokenizer
                    
        
    def __len__(self):
        return len(self.source)
    

    def __getitem__(self, idx):

        if self.share_tokenizer:
            x, x_len = self.src_tokenizer.transform(self.source[idx], as_array=False, bos=True, eos=True)
            y, y_len = self.src_tokenizer.transform(self.target[idx], as_array=False, bos=True, eos=True)
        else:
            x, x_len = self.src_tokenizer.transform(self.source[idx], as_array=False, bos=True, eos=True)
            y, y_len = self.tgt_tokenizer.transform(self.target[idx], as_array=False, bos=True, eos=True)

        return x[0], y[0], x_len[0], y_len[0]
