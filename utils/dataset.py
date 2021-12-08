import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


class LatentDataset(data.Dataset):
    def __init__(self, code_dir, label_dir, is_Train=True):
        latentcode = np.load(code_dir)
        latentlabel = np.load(label_dir)
        train_len = (int)(0.9 * len(latentcode))
        if is_Train:
            self.code = latentcode[:train_len]
            self.label = latentlabel[:train_len]
        else:
            self.code = latentcode[train_len:]
            self.label = latentlabel[train_len:]

    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        return self.code[idx], self.label[idx]
