from typing import Tuple, Dict, List, Callable 

import numpy as np
import torch
from torch.utils.data import Dataset
from tensorflow import keras
from eunjeon import Mecab

import config
from preprocessor import load_data

class ChatbotDataset(Dataset):
    def __init__(self, type: bool, vocab: Dict[str, int], maxlen: int) -> None:
    #def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]]=None) -> None:
        """

        :param type: True -> train, False -> eval
        :param vocab: token2idx
        :param maxlen: vocab max length
        """
        tr_input, tr_label, val_input, val_label = load_data()
        
        if type == True:
            question = tr_input
            answer = tr_label
        else:
            question = val_input
            answer = val_label

        self._corpus = question
        self._label = answer
        self._vocab = vocab
        self._tokenizer = Mecab.pos
        self._pad = keras.preprocessing.sequence.pad_sequences
        self._maxlen = maxlen

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        enc_input = torch.tensor(self.transform([self._corpus[idx].lower()]))
        dec_input, dec_output = torch.tensor(self.transform([self._label[idx].lower()], add_start_end_token=True))

        return enc_input[0], dec_input[0], dec_output[0]
        '''

        input_token = Mecab().pos(self._corpus[idx].lower())
        input_token =[str(pos[0]) + '/' + str(pos[1]) for pos in input_token]
        enc_input = [[self.token2idx(token) for token in input_token]]
        pad_enc_input = torch.tensor(np.array(self._pad(enc_input, value=0, padding='post', truncating='post', maxlen=self._maxlen)))

        output_token = Mecab().pos(self._label[idx].lower())
        output_token =[str(pos[0]) + '/' + str(pos[1]) for pos in output_token]
        dec_input = [[self.token2idx(token) for token in ["<s>"] + output_token]]
        dec_output = [[self.token2idx(token) for token in output_token + ["</s>"]]]
        pad_dec_input = torch.tensor(np.array(self._pad(dec_input, value=0, padding='post', truncating='post', maxlen=self._maxlen)))
        pad_dec_output = torch.tensor(np.array(self._pad(dec_output, value=0, padding='post', truncating='post', maxlen=self._maxlen)))

        return pad_enc_input[0], pad_dec_input[0], pad_dec_output[0]
    
    def token2idx(self, token):
        try:
            return self._vocab[token]
        except:
            return self._vocab["<unk>"]

'''
    def transform(self, str_batch, add_start_end_token=False):
        tokens_batch = self.str2token(str_batch)
        if add_start_end_token is True:
            dec_input = self.token2idx([["<s>"] + tokens for tokens in tokens_batch])
            dec_output = self.token2idx([tokens + ["</s>"] for tokens in tokens_batch])
            pad_dec_input = self._pad(dec_input)
            pad_dec_output = self._pad(dec_output)
            return pad_dec_input, pad_dec_output
        else:
            idxs_batch = self.token2idx(tokens_batch)
            pad_idxs_batch = self._pad(idxs_batch)
            return pad_idxs_batch
    
    def str2token(self, str_batch):
        output = []
        tokens_batch = [Mecab().pos(string) for string in str_batch]
        for tokens in tokens_batch:
            output.append([str(pos[0]) + '/' + str(pos[1]) for pos in tokens])
        return output

    def token2idx(self, tokens_batch):
        idxs_batch = []
        for tokens in tokens_batch:
            idxs_batch.append([self._vocab[token] for token in tokens])
        return idxs_batch


'''
