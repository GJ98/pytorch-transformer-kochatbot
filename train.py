from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
from tqdm import tqdm
import os
import json
import numpy as np

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from evaluate import evaluate, acc
from data_utils.preprocessor import load_vocabulary
from model.net import Transformer
from model.optim import GradualWarmupScheduler

import config
from data_utils.utils import CheckpointManager, SummaryManager
from data_utils.chatbot_dataset import ChatbotDataset

def main():

    # Vocab & Tokenizer
    with open(config.DATA["vocab_path"], mode='rb') as io:
        token2idx = json.load(io)
    # token2idx, idx2token, vocab_size = load_vocabulary()
    config.MODEL['vocab_size'] = len(token2idx)

    # Model & Model Params
    model = Transformer(config=config.MODEL, vocab=token2idx)

    # Train & Val Datasets
    tr_ds = ChatbotDataset(True, token2idx, config.MODEL['maxlen'])
    tr_dl = DataLoader(tr_ds, batch_size=config.MODEL['batch_size'], shuffle=True, num_workers=4, drop_last=False)

    val_ds = ChatbotDataset(False, token2idx, config.MODEL['maxlen'])
    val_dl = DataLoader(val_ds, batch_size=config.MODEL['batch_size'], shuffle=True, num_workers=4, drop_last=False)

    # loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=token2idx["<pad>"]) # nn.NLLLoss()

    # optim
    # torch.optim.SGD(params=model.parameters(), lr=model_config.learning_rate)
    opt = optim.Adam(params=model.parameters(), lr=config.MODEL['learning_rate']) 

    # scheduler = ReduceLROnPlateau(opt, factor=0.9, patience=10)  # Check
    scheduler = GradualWarmupScheduler(opt, multiplier=8, total_epoch=config.MODEL['epochs'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # save
    checkpoint_manager = CheckpointManager('experiments/base_model')
    summary_manager = SummaryManager('experiments/base_model')
    best_val_loss = 1e+10
    best_train_acc = 0

    # load
    if Path('experiments/base_model/best.tar').exists():
        print("pretrained model exists")
        checkpoint = checkpoint_manager.load_checkpoint('best.tar')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Train
    for epoch in tqdm(range(config.MODEL['epochs']), desc='epoch', total=config.MODEL['epochs']):
        scheduler.step(epoch)
        print("\nepoch : {}, lr: {}".format(epoch, opt.param_groups[0]['lr']))
        tr_loss = 0
        tr_acc = 0
        model.train()

        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            opt.zero_grad()

            enc_input, dec_input, dec_output = map(lambda elm: elm.to(device), mb)
            y_pred = model(enc_input, dec_input)
            y_pred_copy = y_pred.detach()
            dec_output_copy = dec_output.detach()

            # loss 계산을 위해 shape 변경
            y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_output.view(-1).long()

            # padding 제외한 value index 추출
            real_value_index = [dec_output != 0]

            # padding은 loss 계산시 제외
            mb_loss = loss_fn(y_pred[real_value_index], dec_output[real_value_index]) # Input: (N, C) Target: (N)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_pred, dec_output)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()
            tr_loss_avg =  tr_loss / (step + 1)
            tr_acc_avg = tr_acc / (step + 1)
            tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc_avg}
            total_step = epoch * len(tr_dl) + step

            # Eval
            if total_step % config.MODEL['summary_step'] == 0 and total_step != 0:
                model.eval()
                print("eval: ")
                val_summary = evaluate(model, loss_fn, val_dl, device)

                tqdm.write('\nepoch : {}, step : {}, '
                           'tr_loss: {:.3f}, val_loss: {:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1,
                                                                                                       total_step,
                                                                                                       tr_summary['loss'],
                                                                                                       val_summary['loss'], 
                                                                                                       tr_summary['acc'],
                                                                                                       val_summary['acc']))

                val_loss = val_summary['loss']
                # is_best = val_loss < best_val_loss # loss 기준
                is_best = tr_acc > best_train_acc # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)

                # Save
                if is_best:
                    print("[Best model Save] train_acc: {}, train_loss: {}, val_loss: {}".format(tr_summary['acc'], 
                                                                                                 tr_summary['loss'],
                                                                                                 val_loss))
                    # CPU에서도 동작 가능하도록 자료형 바꾼 뒤 저장
                    state = {'epoch': epoch + 1,
                             'model_state_dict': model.to(torch.device('cpu')).state_dict(),
                             'opt_state_dict': opt.state_dict()}
                    summary = {'train': tr_summary, 'validation': val_summary}

                    summary_manager.update(summary)
                    summary_manager.save('summary.json')
                    checkpoint_manager.save_checkpoint(state, 'best.tar')

                    best_val_loss = val_loss

                model.to(device)
                model.train()
            else:
                if step % 50 == 0:
                    print('\nepoch : {}, step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, 
                                                                                          total_step, 
                                                                                          tr_summary['loss'], 
                                                                                          tr_summary['acc']))



if __name__ == '__main__':
    main()