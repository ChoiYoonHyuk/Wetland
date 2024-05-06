import numpy as np
import argparse
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import FAConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB


class fagcn(torch.nn.Module):
    def __init__(self, dim):
        super(fagcn, self).__init__()
        self.fagcn = FAConv(dim, eps=.5, dropout=.5)
        self.pred = nn.Linear(dim, 2)
        
    def forward(self, x, trg_edge):
        trg_x = self.fagcn(x, x, trg_edge)
        trg_out = self.pred(self.fagcn(trg_x, x, trg_edge))
        
        return trg_out
