import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from sklearn.model_selection import StratifiedKFold


class GDataset(Dataset):
    def __init__(self, idxs, csv, feature_mean, feature_std, node_num, feature2_mean, feature2_std, feature3_mean, feature3_std, fold):
        super(GDataset, self).__init__()
        self.fc_matrix_dot = os.path.join('../Data_Preparation')
        self.idxs = idxs
        self.csv = csv
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.node_num = node_num
        self.feature2_mean = feature2_mean
        self.feature2_std = feature2_std
        self.feature3_mean = feature3_mean
        self.feature3_std = feature3_std
        self.fold = fold

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        dir_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(int)
        label = label.repeat(3, axis=0)

        fc_matrix = []
        feature = []
        fc_matrix_2 = []
        feature_2 = []
        fc_matrix_3 = []
        feature_3 = []

        for i in range(1, 4):
            fc_matrix_path = os.path.join(self.fc_matrix_dot, 'Pearson', f'{i}Multiple', str(dir_name), f'fcmatrix_116.npy')
            fc_matrix1 = np.load(fc_matrix_path).astype(np.float32)
            fc_matrix1 = fc_matrix1[:90, :90]
            fc_matrix.append(1 - np.sqrt((1 - fc_matrix1) / 2))

            feature_path = os.path.join('../Pretrain/NodeFeature_pretrain/nodeEmbed_formLSTMpretrain', f'{self.fold}', f'{i}Multiple', str(dir_name), f'feature_{self.node_num}.npy')
            feature1 = np.load(feature_path).astype(np.float32)
            feature1 = feature1[:90, :]
            feature.append(((feature1 - self.feature_mean) / self.feature_std).astype(np.float32)[:, :])


            fc_matrix_path = os.path.join(self.fc_matrix_dot, 'Spearman', f'{i}Multiple', str(dir_name), f'fcmatrix_116.npy')
            fc_matrix2 = np.load(fc_matrix_path).astype(np.float32)
            fc_matrix2 = fc_matrix2[:90, :90]
            fc_matrix_2.append(1 - np.sqrt((1 - fc_matrix2) / 2))

            feature_path = os.path.join('../Pretrain/NodeFeature_pretrain/nodeEmbed_formLSTMpretrain', f'{self.fold}', f'{i}Multiple', str(dir_name), f'feature_{self.node_num}.npy')
            feature2 = np.load(feature_path).astype(np.float32)
            feature2 = feature2[:90, :]
            feature_2.append(((feature2 - self.feature2_mean) / self.feature2_std).astype(np.float32)[:, :])


            fc_matrix_path = os.path.join(self.fc_matrix_dot, 'Pcor', f'{i}Multiple', str(dir_name), f'fcmatrix_116.npy')
            fc_matrix3 = np.load(fc_matrix_path).astype(np.float32)
            fc_matrix3 = fc_matrix3[:90, :90]
            fc_matrix_3.append(1 - np.sqrt((1 - fc_matrix3) / 2))

            feature_path = os.path.join('../Pretrain/NodeFeature_pretrain/nodeEmbed_formLSTMpretrain', f'{self.fold}', f'{i}Multiple', str(dir_name), f'feature_{self.node_num}.npy')
            feature3 = np.load(feature_path).astype(np.float32)
            feature3 = feature3[:90, :]
            feature_3.append(((feature3 - self.feature3_mean) / self.feature3_std).astype(np.float32)[:, :])

        return torch.tensor(np.array(fc_matrix)), torch.tensor(np.array(feature)), torch.tensor(np.array(label)), torch.tensor(np.array(fc_matrix_2)), torch.tensor(np.array(feature_2)), torch.tensor(np.array(fc_matrix_3)), torch.tensor(np.array(feature_3))


def get_data_loader(i_fold, class_name, node_num):

    df = pd.read_csv('../Data_Preparation/NYU5fold.csv')

    if class_name in ['one_vs_two']:
        extrelabel = 0
        df = df[df['label'] != extrelabel]
        df['label'] = df['label'] - 1
    elif class_name in ['zero_vs_two']:
        extrelabel = 1
        df = df[df['label'] != extrelabel]
        df.loc[df['label'] == 2, 'label'] = df.loc[df['label'] == 2, 'label'] - 1
    elif class_name in ['zero_vs_one']:
        extrelabel = 2
        df = df[df['label'] != extrelabel]

    df = df.reset_index()
    train_idxs = np.where(df['fold'] != i_fold)[0]
    test_idxs = np.where(df['fold'] == i_fold)[0]
    # train_idxs = np.load(f'../Data_Preparation/train_idxs_{i_fold}.npy')
    # val_idxs = np.load(f'../Data_Preparation/val_idxs_{i_fold}.npy')

    ## featuremean featurestd to normalization
    feature_list = []
    feature2_list = []
    feature3_list = []
    dftemp = df.iloc[train_idxs]
    for i, row in dftemp.iterrows():
        name = row['name']
        for j in range(1, 4):
            feature_path = os.path.join('../Pretrain/NodeFeature_pretrain', 'nodeEmbed_formLSTMpretrain', f'{i_fold}', f'{j}Multiple', str(name), f'feature_{node_num}.npy')
            feature = np.load(feature_path).astype(np.float32)
            feature = feature[:90, :]
            feature_list.append(feature)

            feature_path2 = os.path.join('../Pretrain/NodeFeature_pretrain', 'nodeEmbed_formLSTMpretrain', f'{i_fold}', f'{j}Multiple', str(name), f'feature_{node_num}.npy')
            feature2 = np.load(feature_path2).astype(np.float32)
            feature2 = feature2[:90, :]
            feature2_list.append(feature2)

            feature_path3 = os.path.join('../Pretrain/NodeFeature_pretrain', 'nodeEmbed_formLSTMpretrain', f'{i_fold}', f'{j}Multiple', str(name), f'feature_{node_num}.npy')
            feature3 = np.load(feature_path3).astype(np.float32)
            feature3 = feature3[:90, :]
            feature3_list.append(feature3)

    print(np.array(feature_list).shape)
    feature_mean = np.array(np.concatenate(feature_list).mean(axis=0))
    feature_std = np.array(np.concatenate(feature_list).std(axis=0))
    feature2_mean = np.array(np.concatenate(feature2_list).mean(axis=0))
    feature2_std = np.array(np.concatenate(feature2_list).std(axis=0))
    feature3_mean = np.array(np.concatenate(feature3_list).mean(axis=0))
    feature3_std = np.array(np.concatenate(feature3_list).std(axis=0))
    TrainDataset = GDataset(train_idxs, df, feature_mean, feature_std, node_num, feature2_mean, feature2_std, feature3_mean, feature3_std, i_fold)
    ValDataset = GDataset(test_idxs, df, feature_mean, feature_std, node_num, feature2_mean, feature2_std, feature3_mean, feature3_std, i_fold)
    TrainLoader = DataLoader(TrainDataset, batch_size=4, shuffle=True)
    ValLoader = DataLoader(ValDataset, batch_size=1)
    return TrainLoader, ValLoader


if __name__ == '__main__':
    TrainLoader, ValLoader = get_data_loader(0, 'zero_vs_one', 16)
    for matrix, feature, crs,a,b,c,d in TrainLoader:
        print(matrix.shape, feature.shape, crs.shape)
        break
    print(' ')
    for matrix, feature, crs,a,b,c,d in ValLoader:
        print(matrix.shape, feature.shape, crs.shape)
        break
