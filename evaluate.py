from tqdm import tqdm
import torch

def evaluate(model, loss_fn, data_loader, device):
    if model.training:
        model.eval()

    summary = {'loss': 0, 'acc': 0}
    num_correct_elms = 0

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        enc_input, dec_input, dec_output = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_pred = model(enc_input, dec_input)

            y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_output.view(-1).long()

            # acc 
            _correct_sum, _num_correct_elms = correct_sum(y_pred, dec_output)
            summary['acc'] += _correct_sum
            num_correct_elms += _num_correct_elms

            # loss
            summary['loss'] += loss_fn(y_pred, dec_output).item() #* dec_output.size()[0]

    # acc
    summary['acc'] /= num_correct_elms

    # loss
    summary['loss'] /= len(data_loader.dataset)

    return summary

def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float()[y != 0].mean() # padding은 acc에서 제거
    return acc

def correct_sum(y_pred, dec_output):
    with torch.no_grad():
        y_pred = y_pred.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
        correct_elms = (y_pred == dec_output).float()[dec_output != 0]
        correct_sum = correct_elms.sum().to(torch.device('cpu')).numpy()  # padding은 acc에서 제거
        num_correct_elms = len(correct_elms)
    return correct_sum, num_correct_elms



