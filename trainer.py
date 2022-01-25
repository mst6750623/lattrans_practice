import torch
import torch.nn as nn
import numpy as np
from pixel2style2pixel.models.stylegan2.model import Generator
from pixel2style2pixel.models.psp import get_keys
from net.transformer import TNet
from net.classifier import classifier


class Trainer(nn.Module):
    def __init__(self, attr_id, label_file):
        super().__init__()
        #to complement
        self.transformer = TNet(mapping_layers=18,
                                mapping_fmaps=512,
                                mapping_lrmul=1,
                                mapping_nonlinearity='lrelu')
        self.classifier = classifier(fmaps=[6048, 2048, 512, 40], activ='relu')
        self.generator = Generator(1024, 512, 8)
        self.attr_id = attr_id
        self.label_file = label_file

    def get_correlation(self):
        lbl = np.load(self.label_file)

        print(lbl.shape)
        correlation = np.corrcoef(lbl)

    def initialize(self, generator_arg, classifier_arg):
        return
        generator_state = torch.load(generator_arg, map_location='cpu')
        self.generator.load_state_dict(get_keys(generator_state, 'encoder'),
                                       strict=True)
        self.classifier.load_state_dict(torch.load(classifier_arg))

    def cal_loss(self, w):
        k = self.attr_id
        p_k = self.classifier(self.transformer(w))[k]
        l_cls = -torch.log(p_k) - ()

        l_reg = torch.MSELoss(w, )
