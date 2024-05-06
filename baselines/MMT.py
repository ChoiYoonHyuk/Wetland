import numpy as np
import argparse
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function


class mmt(nn.Module):
    def __init__(self, dim):
        super(mmt, self).__init__()
        self.src_fe = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.shr_fe = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.trg_fe = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(),
        )
        
        self.pred = nn.Linear(32, 2)
        
    def forward(self, src, trg, device):
        src_shr = self.shr_fe(src)
        src = self.src_fe(src) + src_shr
        
        trg_shr = self.shr_fe(trg)
        trg = self.trg_fe(trg) + trg_shr
        
        src_out = self.pred(src)
        trg_out = self.pred(trg)
        
        return src_out, trg_out