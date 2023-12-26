import time, os, argparse, shutil
import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from Pretrain.model.model import LSTMModel

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")


def val_epoch(model, sig, device):
    model.eval()
    with torch.no_grad():
        feature, out = model(sig.to(device))
        feature = feature.squeeze().cpu().numpy()
    return feature, out


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--seed', type=int, default=664, help='Random seed.')  
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

    class_name = 'zero_vs_one'

    score_acc_list = []

    for ifold in range(5):

        net = LSTMModel(inputDim=1, hiddenNum=16, outputDim=2, layerNum=1, cell="LSTM", use_cuda=True).to(DEVICE)
        net.load_state_dict(torch.load(
            f'checkpoint/lstm_pretrain/{class_name}_lr1.0e-031.0e-03_wd0_hidden_num16_fold{ifold}/model_fold{ifold}.pth'))
        net.to(DEVICE)

        data_fold = pd.read_csv('../Data_Preparation/NYU5fold.csv', skip_blank_lines=True)

        df = data_fold.reset_index()
        train_idxs = np.where(df['fold'] != ifold)[0]
        test_idxs = np.where(df['fold'] == ifold)[0]

        feature_list = []
        dftemp = df.iloc[train_idxs]
        for i in range(1, 4):
            for ii, row in dftemp.iterrows():
                name = row['name']
                feature_path = os.path.join('../Data_Preparation', 'Pearson', f'{i}Multiple', str(name),
                                            f'signal_116.npy')
                feature = np.load(feature_path).astype(np.float32)
                feature = feature[:90, :]
                feature_list.append(feature)

        feature_mean = np.array(np.concatenate(feature_list).mean(axis=0))
        feature_std = np.array(np.concatenate(feature_list).std(axis=0))

        pred_list, target_list = [], []

        for i in range(1, 4):
            for index, row in df.iloc[test_idxs].iterrows():
                name = row['name']
                targets = row['label']
                Signal = np.load(f'../Data_Preparation/Pearson/{i}Multiple/' + str(name) + '/signal_116.npy')

                data = Signal[:90, :]
                data = ((data - feature_mean) / feature_std).astype(np.float32)
                data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                feature, out_1 = val_epoch(net, data, DEVICE)

                preds_score = F.softmax(out_1, dim=2).cpu().numpy()
                preds = np.argmax(preds_score, axis=2)
                pred_list.append(preds)
                target_list.append(targets)

        preds = np.concatenate(pred_list)
        preds = mode(preds.transpose())[0][0]
        targets = target_list

        score_acc = accuracy_score(targets, preds)
        score_acc_list.append(score_acc)

    print(f'ACC {np.array(score_acc_list).mean():.3f}±{np.array(score_acc_list).std():.3f}')


