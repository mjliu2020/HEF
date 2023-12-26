import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np


data_fold = pd.read_csv('NYU5fold.csv', skip_blank_lines=True)
node_num = 116

for i in range(1, 4):
    for index, row in data_fold.iterrows():

        name = row['name']
        label = row['label']
        FCMatrix_path = glob(f'./data/FCmatrix_3Multiple/FCmatrix{i}/{i}_NYU_00' + str(name) + '_func_preproc.mat')
        Signal_path = glob(f'./data/ROISignals_3Multiple/ROISignals{i}/{i}_NYU_00' + str(name) + '_func_preproc.mat')

        FCMatrix = loadmat(FCMatrix_path[0])['cc_matrix'][:node_num, :node_num]
        ROISignal = loadmat(Signal_path[0])['ROISignals'][:, :]

        os.makedirs(os.path.join(f'PearsonNew/{i}Multiple/', str(name)), exist_ok=True)
        np.save(os.path.join(f'PearsonNew/{i}Multiple/', str(name), f'fcmatrix_{node_num}.npy'), FCMatrix)
        np.save(os.path.join(f'PearsonNew/{i}Multiple/', str(name), f'signal_{node_num}.npy'), ROISignal)
