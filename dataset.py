import torch
import torch.utils.data as data
import numpy as np

class LatentDataset(data.Dataset):
    def __init__(self, data_anno_path='./data/celebahq_anno.npy', data_label_path='./data/celebahq_label.npy',isTrain=True):
        self.data = np.load(data_anno_path)
        self.label = np.load(data_label_path)
        train_len = int(0.9*len(self.data))
        if isTrain:
            self.data = self.data[:train_len]
            self.label = self.label[:train_len]
        else:
            self.data = self.data[train_len:]
            self.label = self.label[train_len:]
        
        self.length = len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])

    def __len__(self):
        return self.length



if __name__ == '__main__':
    latent_path = './data/celebahq_dlatents_psp.npy'
    label_path = './data/celebahq_anno.npy'
    dataset = LatentDataset(latent_path, label_path, True)
