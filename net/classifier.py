import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self,fmaps=[6048, 2048, 512, 40],activ='relu'):
        super(classifier, self).__init__()
        self.net = nn.ModuleList()
        for i in range(fmaps)-1:
            in_dim = fmaps[i]
            out_dim = fmaps[i+1]
            self.net.append(nn.Linear(in_dim, out_dim,bias=True))

        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)

    def forward(self,x):
        for layer in self.net[:-1]:
            x = self.relu(layer(x))
        x = self.net[-1](x)
        return x