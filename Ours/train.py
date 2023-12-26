import time, os, argparse, shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from Ours.GraphDataset import get_data_loader
from Ours.net_utils import train_epoch, val_epoch
from Ours.model.model import GAN

from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime

np.seterr(divide='ignore', invalid='ignore')
import warnings
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


def train(i_fold, DEVICE):

    weight_decay = 0
    lr = 1e-3
    lr_decay = 1e-3
    train_epochs = 200
    node_num = 116

    feat_dim = 16
    hidden_num = 8
    class_num = 2
    class_name = 'zero_vs_one'

    network_name = f'{class_name}_lr{lr:.1e}{lr_decay:.1e}_epoch{train_epochs}_wd{weight_decay}' \
                   f'_feature{feat_dim}_hidden_num{hidden_num}_fold{i_fold}'

    save_path = f'result/result1/{network_name}'
    os.makedirs(save_path, exist_ok=True)
    code_path = 'result/result1/code'
    os.makedirs(code_path, exist_ok=True)

    shutil.copy('./train.py', code_path)
    shutil.copy('./GraphDataset.py', code_path)
    shutil.copy('./tools.py', code_path)
    shutil.copy('./net_utils.py', code_path)
    shutil.copy('./model/model.py', code_path)
    shutil.copy('./model/calss.py', code_path)

    writer1 = SummaryWriter('./result/result1/train')
    writer2 = SummaryWriter('./result/result1/val')
    writer3 = SummaryWriter('./result/result1/cp_loss')
    writer5 = SummaryWriter('./result/result1/mse_loss')

    best_val_loss, is_best_loss, best_loss_epoch = 2 ** 20, False, 0
    best_score_acc, is_best_score, best_score_epoch = -2 ** 20, False, 0
    history = pd.DataFrame()

    def save_model(model, is_best_loss=False, is_best_score=False):

        model_state = {'state_dict': model.state_dict(), 'epoch': epoch}

        model_path = os.path.join(save_path, f"{network_name}_fold{i_fold}.pth")
        torch.save(model_state, model_path)
        if is_best_loss:
            best_model_path = os.path.join(save_path, f"{network_name}_fold{i_fold}_best_loss.pth")
            shutil.copy(model_path, best_model_path)
        if is_best_score:
            best_model_path = os.path.join(save_path, f"{network_name}_fold{i_fold}_best_score.pth")
            shutil.copy(model_path, best_model_path)

    TrainLoader, ValLoader = get_data_loader(i_fold, class_name, 16)

    model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, eta_min=lr_decay)

    logging.info(ifold)
    setting = f'weight_decay-{weight_decay}, lr-{lr}, lr_decay-{lr_decay}, train_epochs-{train_epochs}'
    logging.info(setting)

    for epoch in range(train_epochs):
        cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc, loss1, loss3 = train_epoch(model, TrainLoader, DEVICE, optimizer)

        val_loss, val_acc, score_auc, preds, preds_score, targets, f1, precision, recall, specificity, loss4, loss6 = val_epoch(
            model, ValLoader, DEVICE)

        is_best_loss, is_best_score = val_loss < best_val_loss, val_acc > best_score_acc
        best_val_loss, best_score_acc = min(val_loss, best_val_loss), max(val_acc, best_score_acc)

        lr_scheduler.step()

        _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'cp_loss': [loss1], 'mse_loss': [loss3], \
             'train_acc': [train_acc], 'val_loss': [val_loss], 'valcp_loss': [loss4], \
             'valmse_loss': [loss6], \
             'val_acc': [val_acc]})
        history = history._append(_h, ignore_index=True)

        msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, cp_loss:{loss1:.4f}, mse_loss:{loss3:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, valcp_loss:{loss4:.4f}, mse_loss:{loss6:.4f},  val_acc:{val_acc:.4f}"
        if is_best_loss:
            best_loss_epoch, msg = epoch, msg
        if is_best_score:
            best_score_epoch, msg = epoch, msg
            best_score_auc = score_auc
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_specificity = specificity
            best_targets = targets
            best_preds = preds

        logging.info(msg)
        save_model(model, is_best_loss, is_best_score)

    train_acc1 = history['train_acc'].dropna()
    val_acc1 = history['val_acc'].dropna()
    train_loss1 = history['train_loss'].dropna()
    train_loss2 = history['cp_loss'].dropna()
    train_loss4 = history['mse_loss'].dropna()
    val_loss1 = history['val_loss'].dropna()
    val_loss2 = history['valcp_loss'].dropna()
    val_loss4 = history['valmse_loss'].dropna()
    for i in range(np.array(train_loss1).shape[0]):
        writer1.add_scalar(f'loss_sum/ifold{i_fold}', np.array(train_loss1)[i], i)
        writer2.add_scalar(f'loss_sum/ifold{i_fold}', np.array(val_loss1)[i], i)
        writer1.add_scalar(f'cp_loss/ifold{i_fold}', np.array(train_loss2)[i], i)
        writer1.add_scalar(f'mse_loss/ifold{i_fold}', np.array(train_loss4)[i], i)
        writer1.add_scalar(f'acc/ifold{i_fold}', np.array(train_acc1)[i], i)
        writer2.add_scalar(f'acc/ifold{i_fold}', np.array(val_acc1)[i], i)
        writer2.add_scalar(f'cp_loss/ifold{i_fold}', np.array(val_loss2)[i], i)
        writer2.add_scalar(f'mse_loss/ifold{i_fold}', np.array(val_loss4)[i], i)
        writer3.add_scalar(f'val_loss_3/ifold{i_fold}', np.array(val_loss2)[i], i)
        writer5.add_scalar(f'val_loss_3/ifold{i_fold}', np.array(val_loss4)[i], i)
        writer3.add_scalar(f'train_loss_3/ifold{i_fold}', np.array(train_loss2)[i], i)
        writer5.add_scalar(f'train_loss_3/ifold{i_fold}', np.array(train_loss4)[i], i)

    return best_score_acc, best_score_auc, best_f1, best_precision, best_recall, best_specificity, best_targets, best_preds


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='gpu')
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
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

    accs = []
    aucs = []
    f1 = []
    precision = []
    recall = []
    specificity = []
    targets = []
    preds = []
    open_log(f'result/result1')

    for ifold in [0,1,2,3,4]:

        best_score_acc, best_score_auc, best_f1, best_precision, best_recall, best_specificity, best_targets, best_preds = train(
            ifold, DEVICE)
        accs.append(best_score_acc)
        aucs.append(best_score_auc)
        f1.append(best_f1)
        precision.append(best_precision)
        recall.append(best_recall)
        specificity.append(best_specificity)
        targets.extend(best_targets)
        preds.extend(best_preds)

    logging.info(accs)
    logging.info(aucs)
    logging.info(f1)
    logging.info(precision)
    logging.info(recall)
    logging.info(specificity)
    logging.info(targets)
    logging.info(preds)

    logging.info(f'ACC {np.array(accs).mean():.3f}±{np.array(accs).std():.3f}')
    logging.info(f'AUC {np.array(aucs).mean():.3f}±{np.array(aucs).std():.3f}')
    logging.info(f'F1 {np.array(f1).mean():.3f}±{np.array(f1).std():.3f}')
    logging.info(f'precision {np.array(precision).mean():.3f}±{np.array(precision).std():.3f}')
    logging.info(f'recall {np.array(recall).mean():.3f}±{np.array(recall).std():.3f}')
    logging.info(f'specificity {np.array(specificity).mean():.3f}±{np.array(specificity).std():.3f}')


