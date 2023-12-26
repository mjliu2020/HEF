import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm as tqdmauto
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.stats import mode


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


def train_epoch(model, loader, device, optimizer, verbose=False):
    model.train()
    train_losses = []
    train_losses1, train_losses3 = [], []
    pred_list, target_list = [], []
    optimizer.zero_grad()
    progress_bar = tqdmauto(loader) if verbose else None

    for batch_idx, (fc_matrixs, feature, targets, adj2, x2, adj3, x3) in enumerate(loader):

        sizeindex = targets.shape[0]
        fc_matrixs = fc_matrixs.reshape(-1, 90, 90)
        feature = feature.reshape(-1, 90, 16)
        targets = torch.squeeze(targets.reshape(-1, 1))
        adj2 = adj2.reshape(-1, 90, 90).to(device)
        x2 = x2.reshape(-1, 90, 16).to(device)
        adj3 = adj3.reshape(-1, 90, 90).to(device)
        x3 = x3.reshape(-1, 90, 16).to(device)

        fc_matrixs_batch = fc_matrixs.to(device)
        feature_batch = feature.to(device)
        targets_batch = torch.as_tensor(targets, dtype=torch.long).to(device)
        preds, out = model(feature_batch, fc_matrixs_batch, x2, adj2, x3, adj3)

        cosloss = 0
        for i in range(sizeindex):
            index1 = [0, 1]
            index2 = [0, 2]
            index3 = [1, 2]
            cosine_loss = torch.nn.MSELoss(reduction='mean')  # nn.CosineEmbeddingLoss(margin=0.2)
            cosloss = cosine_loss(out[(index1[0] + i * 3), :], out[(index1[1] + i * 3), :]) + cosloss
            cosloss = cosine_loss(out[(index2[0] + i * 3), :], out[(index2[1] + i * 3), :]) + cosloss
            cosloss = cosine_loss(out[(index3[0] + i * 3), :], out[(index3[1] + i * 3), :]) + cosloss
        cosloss = cosloss / (sizeindex * 3)

        weight = torch.cuda.FloatTensor([98 / 73, 1])
        lossfunction = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        loss1 = lossfunction(preds, targets_batch)
        # loss1 = F.cross_entropy(preds, targets_batch)
        loss = 1*loss1 + 1*cosloss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_np = loss.detach().cpu().item()
        train_losses.append(loss_np)
        loss1_np = loss1.detach().cpu().item()
        train_losses1.append(loss1_np)
        loss3_np = cosloss.detach().cpu().item()
        train_losses3.append(loss3_np)

        preds_score = F.softmax(preds, dim=1).detach().cpu().numpy()
        preds = np.argmax(preds_score, axis=1)
        pred_list.append(preds)
        targets = targets.detach().cpu().numpy()
        target_list.append(targets)

        if verbose:
            progress_bar.set_postfix_str(f"loss: {loss_np:.4f}, smooth_loss: {np.mean(train_losses[-20:]):.4f}")
            progress_bar.update(1)
    if verbose:
        progress_bar.close()

    preds = np.concatenate(pred_list)
    targets = np.concatenate(target_list)
    score_acc = accuracy_score(targets, preds)
    return np.asarray(train_losses).mean(), score_acc, np.asarray(train_losses1).mean(), 1*np.asarray(train_losses3).mean()


def val_epoch(model, loader, device):
    model.eval()
    val_losses = []
    test_losses1, test_losses3 = [], []
    pred_list, target_list = [], []
    preds_score_list = []
    with torch.no_grad():
        for fc_matrixs, feature, targets, adj2, x2, adj3, x3 in loader:

            sizeindex = targets.shape[0]
            fc_matrixs = fc_matrixs.reshape(-1, 90, 90)
            feature = feature.reshape(-1, 90, 16)
            targets = torch.squeeze(targets.reshape(-1, 1))
            adj2 = adj2.reshape(-1, 90, 90).to(device)
            x2 = x2.reshape(-1, 90, 16).to(device)
            adj3 = adj3.reshape(-1, 90, 90).to(device)
            x3 = x3.reshape(-1, 90, 16).to(device)
            fc_matrixs_batch = fc_matrixs.to(device)
            feature_batch = feature.to(device)
            targets_batch = torch.as_tensor(targets, dtype=torch.long).to(device)

            preds, out = model(feature_batch, fc_matrixs_batch, x2, adj2, x3, adj3)

            cosloss = 0
            for i in range(sizeindex):

                index1 = [0, 1]
                index2 = [0, 2]
                index3 = [1, 2]
                cosine_loss = torch.nn.MSELoss(reduction='mean')  # nn.CosineEmbeddingLoss(margin=0.2)
                cosloss = cosine_loss(out[(index1[0] + i * 3), :], out[(index1[1] + i * 3), :]) + cosloss
                cosloss = cosine_loss(out[(index2[0] + i * 3), :], out[(index2[1] + i * 3), :]) + cosloss
                cosloss = cosine_loss(out[(index3[0] + i * 3), :], out[(index3[1] + i * 3), :]) + cosloss
            cosloss = cosloss / (sizeindex * 3)

            weight = torch.cuda.FloatTensor([98 / 73, 1])
            lossfunction = nn.CrossEntropyLoss(reduction='mean', weight=weight)
            loss1 = lossfunction(preds, targets_batch)
            # loss1 = F.cross_entropy(preds, targets_batch)
            loss = 1*loss1 + 1*cosloss

            loss_np = loss.detach().cpu().item()
            val_losses.append(loss_np)
            loss1_np = loss1.detach().cpu().item()
            test_losses1.append(loss1_np)
            loss3_np = cosloss.detach().cpu().item()
            test_losses3.append(loss3_np)

            preds_score = F.softmax(preds, dim=1).cpu().numpy()
            preds = np.argmax(preds_score, axis=1)
            targets = targets.cpu().numpy()
            preds_score_list.append(preds_score)
            pred_list.append(preds)
            target_list.append(targets)

    targets = np.concatenate(target_list)
    preds_score2 = np.concatenate(preds_score_list)
    preds = np.concatenate(pred_list)

    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    specificity, test = spe(confusion_matrix(targets, preds))
    score_acc = accuracy_score(targets, preds)
    score_auc = roc_auc_score(targets, preds_score2[:, 1])
    return np.asarray(val_losses).mean(), score_acc, score_auc, preds, preds_score, targets, f1, precision, recall, specificity, np.asarray(test_losses1).mean(), 1*np.asarray(test_losses3).mean()
