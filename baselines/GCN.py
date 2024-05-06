import numpy as np
import argparse
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB


class gcn(torch.nn.Module):
    def __init__(self, dim):
        super(gcn, self).__init__()
        self.gcn_1 = GCNConv(dim, 32)
        self.gcn_2 = GCNConv(32, 2)
        
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.gcn_1(x, edge_index)))
        x_out = self.gcn_2(x, edge_index)
        
        return x_out
