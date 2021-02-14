import os
import pandas as pd
from typing import Tuple, Dict, List 

from sklearn.model_selection import train_test_split
from eunjeon import Mecab

import config

def load_data() -> Tuple[List[str], List[str], List[str], List[str]]:

    data_df = pd.read_csv(config.DATA['data_path'], header=0)

    question, answer = list(data_df['Q']), list(data_df['A'])

    tr_input, val_input, tr_label, val_label = train_test_split(question, answer, test_size=0.33,
                                                                                  random_state=42)
    return tr_input, tr_label, val_input, val_label

def load_vocabulary() -> Tuple[Dict[str, int], Dict[int, str], int]:

    vocab_list = []

    if(not (os.path.exists(config.DATA['vocab_path']))):
        tokens = []
        if(os.path.exists(config.DATA['data_path'])):
            data_df = pd.read_csv(config.DATA['data_path'], encoding='utf-8')
            total_data = list(data_df['Q']) + list(data_df['A'])
            for data in total_data:
                print(data)
                tokens += Mecab().pos(data)
            tokens = list(set(tokens))
            tokens = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens]
            tokens = config.SPECIAL_TOKENS + tokens

        with open(config.DATA['vocab_path'], 'w', encoding='utf-8') as vocab_file:
            for token in tokens:
                vocab_file.write(token + '\n')

    with open(config.DATA['vocab_path'], 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            vocab_list.append(line.strip())

    token2idx = {token: idx for idx, token in enumerate(vocab_list)}
    idx2token = {idx: token for idx, token in enumerate(vocab_list)}
    return token2idx, idx2token, len(token2idx)

