import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x


class PositionEncoding(nn.Module):
    
    def __init__(self, model_dim, max_seq_len=80):
        super().__init__()
        self.model_dim = model_dim

        pe = torch.zeros(max_seq_len, model_dim)
        for pos in range(max_seq_len):
            for i in range(0, model_dim, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/model_dim)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/model_dim)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        x = x * math.sqrt(self.model_dim)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x