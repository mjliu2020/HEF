import time, os, argparse, shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import warnings
import heapq
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from Pretrain.pretrain_Dataset import get_data_loader
from Pretrain.pretrain_net_utils import train_epoch, val_epoch
from Pretrain.model.model import LSTMModel

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")


def open_log(log_path, name='train'):
    log_savepath = os.path.join(log_path, name)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    log_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isfile(os.path.join(log_savepath, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.log'.format(log_name)))


# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=logFilename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(i_fold, DEVICE, class_name = '012'):
    weight_decay = 0
    lr = 1e-3
    lr_decay = 1e-3
    train_epochs = 200

    hidden_num = 16
    patience = 200
    cnt_wait = 0
    best = 1e-9

    network_name = f'{class_name}_lr{lr:.1e}{lr_decay:.1e}_wd{weight_decay}' \
                   f'_hidden_num{hidden_num}_fold{i_fold}'

    save_path = f'result/lstm_pretrain/{network_name}'
    os.makedirs(save_path, exist_ok=True)
    code_path = 'result/lstm_pretrain/code'
    os.makedirs(code_path, exist_ok=True)

    shutil.copy('./pretrain_train.py', code_path)
    shutil.copy('./pretrain_Dataset.py', code_path)
    shutil.copy('./pretrain_net_utils.py', code_path)
    shutil.copy('./model/model.py', code_path)

    writer1 = SummaryWriter('./result/lstm_pretrain/train')

    history = pd.DataFrame()

    TrainLoader, ValLoader = get_data_loader(i_fold)

    model = LSTMModel(inputDim=1, hiddenNum=hidden_num, outputDim=2, layerNum=1, cell="LSTM", use_cuda=True).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, eta_min=lr_decay)

    logging.info(ifold)

    for epoch in range(train_epochs):

        cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, TrainLoader, DEVICE, optimizer)
        val_loss, val_acc, preds, targets, score_acc1_bala = val_epoch(model, ValLoader, DEVICE)

        lr_scheduler.step()

        _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'train_acc': [train_acc], 'val_loss': [val_loss], 'val_acc': [val_acc]})
        history = history._append(_h, ignore_index=True)

        msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}"
        logging.info(msg)

        if val_acc >= best:
            best = val_acc
            best_val_acc = best
            best_val_acc_bala = score_acc1_bala
            best_train_acc = train_acc

            best_targets = targets
            best_preds = preds

            cnt_wait = 0
            model_path = os.path.join(save_path, f"model_fold{i_fold}.pth")
            torch.save(model.state_dict(), model_path)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

    train_loss1 = history['train_loss'].dropna()
    for i in range(np.array(train_loss1).shape[0]):
        writer1.add_scalar(f'loss/ifold{i_fold}', np.array(train_loss1)[i], i)

    return train_loss, best_train_acc, val_loss, best_val_acc, best_targets, best_preds, best_val_acc_bala


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--seed', type=int, default=66, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    open_log(f'result/lstm_pretrain')
    class_name = 'zero_vs_one'

    train_acc_list, val_acc_list, val_acc_bala_list = [], [], []
    targets = []
    preds = []
    for ifold in range(5):

        train_loss, train_acc, val_loss, val_acc, best_targets, best_preds, best_val_acc_bala = train(ifold, DEVICE, class_name)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        val_acc_bala_list.append(best_val_acc_bala)
        targets.extend(best_targets)
        preds.extend(best_preds)

    logging.info(f'train_acc_list {train_acc_list}')
    logging.info(f'val_acc_list {val_acc_list}')
    logging.info(f'val_acc_bala_list {val_acc_bala_list}')
    logging.info(f'train_acc {np.array(train_acc_list).mean():.4f}±{np.array(train_acc_list).std():.4f}')
    logging.info(f'val_acc {np.array(val_acc_list).mean():.4f}±{np.array(val_acc_list).std():.4f}')
    logging.info(f'val_acc_bala {np.array(val_acc_bala_list).mean():.4f}±{np.array(val_acc_bala_list).std():.4f}')


