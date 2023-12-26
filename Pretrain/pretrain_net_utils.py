import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm.auto import tqdm as tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


def spe(confusion_matrix):

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    return np.average(TN/(TN+FP)), np.average(TP/(TP+FP))


def train_epoch(model, loader, device, train_optimizer):

    model.train()
    total_loss, total_num = 0.0, 0
    pred_list, target_list = [], []

    for batch_idx, (pos1, targets) in enumerate(loader):

        batch_size = pos1.shape[0]
        pos1 = pos1.reshape(-1, 90, 175).to(device)
        targets = targets.reshape(-1, 1).to(device)
        targets = torch.as_tensor(targets, dtype=torch.long).to(device)
        targets = targets.repeat(1, 90)

        feature_1, out_1 = model(pos1)

        out_1 = out_1.transpose(2, 1)
        weight = torch.cuda.FloatTensor([98/73, 1])
        lossfunction = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        loss = lossfunction(out_1, targets)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        out_1 = out_1.transpose(2, 1)
        preds_score = F.softmax(out_1, dim=2).detach().cpu().numpy()
        preds = np.argmax(preds_score, axis=2)
        targets = targets.cpu().numpy()

        pred_list.append(preds)
        target_list.append(targets)

    preds1 = np.concatenate(pred_list)
    preds1 = mode(preds1.transpose())[0][0]
    targets1 = np.concatenate(target_list)[:, 0]
    score_acc1 = accuracy_score(targets1, preds1)

    return total_loss / total_num, score_acc1


def val_epoch(model, loader, device):

    model.eval()
    total_loss, total_num = 0.0, 0
    pred_list, target_list = [], []

    with torch.no_grad():
        for pos1, targets in loader:

            batch_size = pos1.shape[0]
            pos1 = pos1.reshape(-1, 90, 175).to(device)
            targets = targets.reshape(-1, 1).to(device)
            targets = torch.as_tensor(targets, dtype=torch.long).to(device)
            targets = targets.repeat(1, 90)

            feature_1, out_1 = model(pos1)

            out_1 = out_1.transpose(2, 1)
            weight = torch.cuda.FloatTensor([98/73, 1])
            lossfunction = nn.CrossEntropyLoss(reduction='mean', weight=weight)
            loss = lossfunction(out_1, targets)

            out_1 = out_1.transpose(2, 1)
            preds_score = F.softmax(out_1, dim=2).cpu().numpy()
            preds = np.argmax(preds_score, axis=2)
            targets = targets.cpu().numpy()
            pred_list.append(preds)
            target_list.append(targets)

            total_num += batch_size
            total_loss += loss.item() * batch_size

        preds1 = np.concatenate(pred_list)
        preds1 = mode(preds1.transpose())[0][0]
        targets1 = np.concatenate(target_list)[:, 0]
        score_acc1 = accuracy_score(targets1, preds1)
        score_acc1_bala = balanced_accuracy_score(targets1, preds1)

        targets2 = targets1.reshape(-1, 3)
        preds2 = preds1.reshape(-1, 3)
        preds2 = mode(preds2.transpose())[0][0]
        targets2 = mode(targets2.transpose())[0][0]

        return total_loss / total_num, score_acc1, preds2, targets2, score_acc1_bala



