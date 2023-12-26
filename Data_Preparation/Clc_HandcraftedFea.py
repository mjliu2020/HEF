import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np


def get_skew_kurt(data):
    data2 = data ** 2  
    data3 = data ** 3  
    niu = data.mean(axis=0)
    niu2 = data2.mean(axis=0)
    niu3 = data3.mean(axis=0)
    sigma = np.sqrt(niu2 - niu * niu)
    niu4 = ((data - data.mean(axis=0)) ** 4).mean(axis=0)
    skew = (niu3 - 3 * niu * sigma ** 2 - niu ** 3) / (sigma ** 3)  
    kurt = niu4 / (sigma ** 4)  
    return niu, sigma, skew, kurt


data_fold = pd.read_csv('NYU5fold.csv', skip_blank_lines=True)
node_num = 116

for i in range(1, 4):
    for index, row in data_fold.iterrows():

        feature_list = []
        name = row['name']
        label = row['label']
        # FCMatrix_path = glob(f'./data/FCmatrix_3Multiple/FCmatrix{i}/{i}_NYU_00' + str(name) + '_func_preproc.mat')
        Signal_path = glob(f'./data/ROISignals_3Multiple/ROISignals{i}/{i}_NYU_00' + str(name) + '_func_preproc.mat')

        # FCMatrix = loadmat(FCMatrix_path[0])['cc_matrix'][:node_num, :node_num]
        ROISignal = loadmat(Signal_path[0])['ROISignals'][:, :]
        ROISignal = ROISignal.T

        fmri_mean, fmri_var = np.mean(ROISignal, axis=0), np.var(ROISignal, axis=0)
        _, _, skew, kurt = get_skew_kurt(ROISignal)
        feature_list.extend([skew, kurt, fmri_mean, fmri_var])

        feature = np.stack(feature_list, axis=-1)

        os.makedirs(os.path.join(f'PearsonNew/{i}Multiple/', str(name)), exist_ok=True)
        os.makedirs(os.path.join(f'SpearmanNew/{i}Multiple/', str(name)), exist_ok=True)
        os.makedirs(os.path.join(f'PcorNew/{i}Multiple/', str(name)), exist_ok=True)
        np.save(os.path.join(f'PearsonNew/{i}Multiple/', str(name), f'feature_{node_num}.npy'), feature)
        np.save(os.path.join(f'SpearmanNew/{i}Multiple/', str(name), f'feature_{node_num}.npy'), feature)
        np.save(os.path.join(f'PcorNew/{i}Multiple/', str(name), f'feature_{node_num}.npy'), feature)

