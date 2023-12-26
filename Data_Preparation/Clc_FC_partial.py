import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy import stats, linalg


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.


    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


data_fold = pd.read_csv('NYU5fold.csv', skip_blank_lines=True)
node_num = 116

for flag in range(1, 4):
    for index, row in data_fold.iterrows():

        name = row['name']
        print(name)
        label = row['label']
        ROISignal_path = glob(f'./data/ROISignals_3Multiple/ROISignals{flag}/{flag}_NYU_00' + str(name) + '_func_preproc.mat')

        ROISignal = loadmat(ROISignal_path[0])['ROISignals'][:, :]
        ROISignal = ROISignal.T

        FCMatrix = partial_corr(ROISignal)
        FCMatrix = np.nan_to_num(FCMatrix)

        os.makedirs(os.path.join(f'PcorNew/{flag}Multiple/', str(name)), exist_ok=True)
        np.save(os.path.join(f'PcorNew/{flag}Multiple/', str(name), f'fcmatrix_{node_num}.npy'), FCMatrix)
        np.save(os.path.join(f'PcorNew/{flag}Multiple/', str(name), f'signal_{node_num}.npy'), ROISignal.T)
