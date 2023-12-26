import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy.stats import spearmanr


data_fold = pd.read_csv('NYU5fold.csv', skip_blank_lines=True)
node_num = 116

for flag in range(1, 4):
    for index, row in data_fold.iterrows():

        name = row['name']
        label = row['label']
        ROISignal_path = glob(f'./data/ROISignals_3Multiple/ROISignals{flag}/{flag}_NYU_00' + str(name) + '_func_preproc.mat')

        ROISignal = loadmat(ROISignal_path[0])['ROISignals'][:, :]
        ROISignal = ROISignal.T

        rho, pval = spearmanr(ROISignal)
        FCMatrix = np.nan_to_num(rho)

        os.makedirs(os.path.join(f'SpearmanNew/{flag}Multiple/', str(name)), exist_ok=True)
        np.save(os.path.join(f'SpearmanNew/{flag}Multiple/', str(name), f'fcmatrix_{node_num}.npy'), FCMatrix)
        np.save(os.path.join(f'SpearmanNew/{flag}Multiple/', str(name), f'signal_{node_num}.npy'), ROISignal.T)
