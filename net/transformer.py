import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#----------------------------------------------------------------------------
# Get weight tensor for a convolution or fully-connected layer.


def get_weight(weight, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(
        weight.size()
        [1:])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init
    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        runtime_coef = he_std * lrmul
    else:
        runtime_coef = lrmul
    return weight * runtime_coef


class DenseLayer():
    def __init__(self, input_dim, output_dim, lrmul, gain=1, use_wscale=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.lrmul = lrmul
        self.gain = gain
        self.use_wscale = use_wscale
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, latents_in):
        w = get_weight(self.weight, self.gain, self.use_wscale, self.lrmul)
        b = self.b
        x = F.linear(latents_in, w, b)
        return x


class TNet(nn.Module):
    def __init__(self,
                 mapping_layers=18,
                 mapping_fmaps=512,
                 mapping_lrmul=1,
                 mapping_nonlinearity='lrelu'):
        super(TNet, self).__init__()
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.dense = nn.ModuleList()
        for layer_idx in range(mapping_layers):
            self.dense.append(
                DenseLayer(mapping_fmaps, mapping_fmaps, mapping_lrmul))

    def forward(self, latents_in, coeff):
        out = []
        for layer_idx in range(self.mapping_layers):
            out.append(self.dense[layer_idx](latents_in[layer_idx]))
        x = torch.cat(out)
        x = latents_in + coeff * x
        return x


if __name__ == "__main__":
    #net = TNet()
    latents_in = torch.randn(1, 18, 512)
    print(torch.Tensor(2, 3))
