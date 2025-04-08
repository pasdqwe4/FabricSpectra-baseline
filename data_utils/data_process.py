import torch
import numpy as np
import os

class NIRDataset_train(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir,  comp_dir):
        data_list = np.load(data_dir)
        data_labels = np.load(label_dir)
        data_comps = np.load(comp_dir)

        '''For hierarchical setting'''
        # dalei_multi = np.zeros(([int(data_labels.shape[0]), 5]))
        # dalei_regress = np.zeros(([int(data_labels.shape[0]), 5]))
        #
        # for i in range(int(data_labels.shape[0])):
        #     if data_labels[i, 0] == 1:
        #         dalei_multi[i, 0] = 1
        #         dalei_regress[i, 0] = data_comps[i, 0]
        #     if data_labels[i, 1] or data_labels[i, 7] or data_labels[i, 10] == 1:
        #         dalei_multi[i, 1] = 1
        #         dalei_regress[i, 1] = data_comps[i, 1] + data_comps[i, 7] + data_comps[i, 10]
        #     if data_labels[i, 2] or data_labels[i, 11] == 1:
        #         dalei_multi[i, 2] = 1
        #         dalei_regress[i, 2] = data_comps[i, 2] + data_comps[i, 11]
        #     if data_labels[i, 6] or data_labels[i, 8] or data_labels[i, 9] == 1:
        #         dalei_multi[i, 3] = 1
        #         dalei_regress[i, 3] = data_comps[i, 6] + data_comps[i, 8] + data_comps[i, 9]
        #     if data_labels[i, 4] or data_labels[i, 5] or data_labels[i, 3] == 1:
        #         dalei_multi[i, 4] = 1
        #         dalei_regress[i, 4] = data_comps[i, 4] + data_comps[i, 5] + data_comps[i, 3]

        #### Note######
        # 0# SP == 0
        # 1# CA, W, S == 1, 7, 10
        # 2# C, L == 2, 11
        # 3# N, A, P == 6, 8, 9
        # 4# R,M, T == 3,4,5
        #################
        # ['SP', 'CA', 'L', 'R', 'M', 'T', 'P', 'W', 'N', 'A', 'S', 'C']
        #   0      1    2    3    4    5    6    7    8    9   10   11

        data_x = data_list.reshape((int(data_labels.shape[0]), 2, 200))

        self.X = data_x
        self.Y = data_labels
        self.comps = data_comps
    def __getitem__(self, item):
        x = self.X[item, :]
        y = self.Y[item, :]
        comps = self.comps[item, :]

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.int64), torch.from_numpy(
            comps).float()
    def __len__(self):
        return torch.tensor(self.X.shape[0], dtype=torch.int64)

class NIRDataset_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, comp_dir):

        data_list = np.load(data_dir)
        data_labels_m = np.load(label_dir)
        data_comps = np.load(comp_dir)

        '''For hierarchical'''
        # dalei_multi = np.zeros(([int(data_labels.shape[0]), 5]))
        # dalei_regress = np.zeros(([int(data_labels.shape[0]), 5]))
        # for i in range(int(data_labels_m.shape[0])):
        #     if data_labels_m[i, 0] == 1:
        #         dalei_multi[i, 0] = 1
        #     if data_labels_m[i, 1] or data_labels_m[i, 7] or data_labels_m[i, 10] == 1:
        #         dalei_multi[i, 1] = 1
        #     if data_labels_m[i, 2] or data_labels_m[i, 11] == 1:
        #         dalei_multi[i, 2] = 1
        #     if data_labels_m[i, 6] or data_labels_m[i, 8] or data_labels_m[i, 9] == 1:
        #         dalei_multi[i, 3] = 1
        #     if data_labels_m[i, 4] or data_labels_m[i, 5] or data_labels_m[i, 3] == 1:
        #     if data_labels[i, 4] or data_labels[i, 5] or data_labels[i, 3] == 1:
        #         dalei_multi[i, 4] = 1
        #         dalei_regress[i, 4] = data_comps[i, 4] + data_comps[i, 5] + data_comps[i, 3]

        data_x = data_list.reshape((int(data_labels_m.shape[0]), 2, 200))

        self.X = data_x
        self.Y = data_labels_m
        self.comps = data_comps

    def __getitem__(self, item):
        x = self.X[item, :]
        y = self.Y[item, :]
        comps = self.comps[item, :]

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.int64), torch.from_numpy(
            comps).float()
    def __len__(self):
        return torch.tensor(self.X.shape[0], dtype=torch.int64)


