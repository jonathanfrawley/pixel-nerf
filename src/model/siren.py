from torch import nn
import torch
import numpy as np

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x, **kwargs):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

class SeqWrapper(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.mods = nn.ModuleList([*modules])
    
    def forward(self, x, **kwargs):
        for layer in self.mods:
            x = layer(x)
        return x

def gon_model(dimensions):
    first_layer = SirenLayer(dimensions[0], dimensions[1], is_first=True)
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
        other_layers.append(SirenLayer(dim0, dim1))
    final_layer = SirenLayer(dimensions[-2], dimensions[-1], is_last=True)
    return SeqWrapper(first_layer, *other_layers, final_layer)

def from_conf(conf, d_in, **kwargs):
    n_blocks = conf.get_int("n_blocks", 5)
    d_hidden = conf.get_int("d_hidden", 128)
    dims = [d_in] + [d_hidden] * n_blocks + [4]
    return gon_model(dims)