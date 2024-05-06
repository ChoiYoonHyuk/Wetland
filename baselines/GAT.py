import numpy as np
import argparse
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB


class gat(torch.nn.Module):
    def __init__(self, dim):
        super(gat, self).__init__()
        self.conv1 = GATConv(dim, 8, heads=4, dropout=0.6)
        self.conv2 = GATConv(32, 2, heads=1, concat=False, dropout=0.6)
        
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.conv1(x, edge_index)))
        x_out = self.conv2(x, edge_index)
        
        return x_out
