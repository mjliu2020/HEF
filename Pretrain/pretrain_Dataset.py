import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from sklearn.model_selection import StratifiedKFold
import heapq


class GDataset(Dataset):
    def __init__(self, idxs, csv, feature_mean, feature_std):
        super(GDataset, self).__init__()
        self.fc_matrix_dot = os.path.join('../Data_Preparation', 'Pearson')
        self.idxs = idxs
        self.csv = csv
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        dir_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(int)
        label = label.repeat(3, axis=0)

        feature_list = []
        for i in range(1, 4):
            feature_path = os.path.join(self.fc_matrix_dot, f'{i}Multiple', str(dir_name), 'signal_116.npy')
            feature = np.load(feature_path).astype(np.float32)
            feature = np.array(feature[:90, :])
            feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)

            feature_list.append(feature)

        return torch.tensor(np.array(feature_list)), np.array(torch.tensor(label))


def get_data_loader(i_fold):

    df = pd.read_csv('../Data_Preparation/NYU5fold.csv')
    df = df.reset_index()

    train_idxs0 = np.where(df['fold'] != i_fold)[0]
    train_idxs1 = train_idxs0
    y = df['label'][np.where(df['fold'] != i_fold)[0]]
    folder = StratifiedKFold(n_splits=5, random_state=12, shuffle=True)
    for train_index, val_index in folder.split(train_idxs1, y):
        train_idxs = train_idxs0[train_index]
        val_idxs = train_idxs0[val_index]
        np.save(f'../Data_Preparation/train_idxs_{i_fold}.npy', train_idxs)
        np.save(f'../Data_Preparation/val_idxs_{i_fold}.npy', val_idxs)
        break
    test_idxs = np.where(df['fold'] == i_fold)[0]

    ## featuremean featurestd to normalization
    feature_list = []
    dftemp = df.iloc[train_idxs]
    for i, row in dftemp.iterrows():
        name = row['name']
        for j in range(1, 4):
            feature_path = os.path.join('../Data_Preparation', 'Pearson', f'{j}Multiple', str(name), f'signal_116.npy')
            feature = np.load(feature_path).astype(np.float32)
            feature = np.array(feature[:90, :])
            feature_list.append(feature)
    print(np.array(feature_list).shape)
    feature_mean = np.array(np.concatenate(feature_list).mean(axis=0))
    feature_std = np.array(np.concatenate(feature_list).std(axis=0))
    TrainDataset = GDataset(train_idxs, df, feature_mean, feature_std)
    ValDataset = GDataset(val_idxs, df, feature_mean, feature_std)
    TrainLoader = DataLoader(TrainDataset, batch_size=16, shuffle=True, drop_last=True)
    ValLoader = DataLoader(ValDataset, batch_size=1)
    return TrainLoader, ValLoader


if __name__ == '__main__':
    TrainLoader, ValLoader = get_data_loader(0)
    for fea, crs in TrainLoader:
        print(fea.shape, crs.shape)
        break
    print(' ')
    for fea, crs in ValLoader:
        print(fea.shape, crs.shape)
        break
