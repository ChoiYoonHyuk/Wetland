import pandas as pd
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torchmetrics import F1Score
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, FAConv
from torch_geometric.utils.sparse import dense_to_sparse
from sklearn.metrics import confusion_matrix
from torch.autograd import Function


class mlp(nn.Module):
    def __init__(self, dim):
        super(mlp, self).__init__()
        self.fe = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 2)
        )

        
    def forward(self, trg):
        out = self.fe(trg)
        
        return out