import torch
import torch.nn as nn
from pixel2style2pixel.models.stylegan2.model import Generator
from pixel2style2pixel.models.psp import get_keys


class Trainer(nn.Module):
    def __init__(self, attr_id):
        super().__init__()
        #to complement
        self.transformer = nn.Sequential()
        self.classifier = nn.Sequential()
        self.generator = Generator(1024, 512, 8)
        self.attr_id = attr_id

    def initialize(self, generator_arg, classifier_arg):
        return
        #generator_state = torch.load(generator_arg, map_location='cpu')
        #self.generator.load_state_dict(get_keys(generator_state, 'encoder'),strict=True)
        #self.classifier.load_state_dict(torch.load(classifier_arg))

    def cal_loss(self, w):
        k = self.attr_id
        p_k = self.classifier(self.transformer(w))[k]
        l_cls = -torch.log(p_k) - ()
