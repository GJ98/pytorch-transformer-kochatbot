from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import sys
import numpy as np
from pathlib import Path

import json
# from konlpy.tag import Okt
from eunjeon import Mecab

import torch
from tensorflow import keras
from model.net import Transformer

import config
from data_utils.utils import Config, CheckpointManager, SummaryManager
# from data_utils.vocab_tokenizer import Tokenizer, Vocabulary, keras_pad_fn, mecab_token_pos_flat_fn

np.set_printoptions(suppress=False)
np.set_printoptions(threshold=sys.maxsize)


def main():

    # Vocab & Tokenizer
    with open(config.DATA['vocab_path'], mode='rb') as io:
        token2idx = json.load(io)
    config.MODEL['vocab_size'] = len(token2idx)
    idx2token = {v:k for k,v in token2idx.items()}

    # Model
    model = Transformer(config=config.MODEL, vocab=token2idx)
    checkpoint_manager = CheckpointManager('experiments/base_model') # experiments/base_model
    checkpoint = checkpoint_manager.load_checkpoint('best.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    while(True):
        input_text = input("문장을 입력하세요: ")
        
        input_token = Mecab().pos(input_text)
        input_token =[str(pos[0]) + '/' + str(pos[1]) for pos in input_token]
        enc_input = [[token2idx[token] for token in input_token]]
        enc_input = keras.preprocessing.sequence.pad_sequences(enc_input, value=0, padding='post', truncating='post', maxlen=config.MODEL['maxlen'])
        enc_input = torch.tensor(enc_input)

        dec_input = torch.tensor([[token2idx["<s>"]]])

        for i in range(config.MODEL['maxlen']):
            y_pred = model(enc_input.to(device), dec_input.to(device))
            y_pred_idx = y_pred.max(dim=-1)[1]
            if (y_pred_idx[0,-1] == token2idx["</s>"]).to(torch.device('cpu')).numpy():
                y_pred_idx = y_pred_idx[0].tolist()
                y_pred_tokens = [idx2token[idx] for idx in y_pred_idx[:-1]]
                pred_str = ''.join([token.split('/')[0] for token in y_pred_tokens])
                print("pred_str: ", pred_str)
                break
            elif i == config.MODEL['maxlen'] - 1:
                y_pred_idx = y_pred_idx[0].tolist()
                y_pred_tokens = [idx2token[idx] for idx in y_pred_idx[:config.MODEL['maxlen']]]
                pred_str = ''.join([token.split('/')[0] for token in y_pred_tokens])
                print("pred_str: ", pred_str)
                break

            dec_input = torch.cat([dec_input.to(torch.device('cpu')), y_pred_idx[0,-1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)

if __name__ == '__main__':

    main()