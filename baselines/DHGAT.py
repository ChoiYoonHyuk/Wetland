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
from torch_geometric.nn import GATConv


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.lambda_ = 1
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = 1
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None
        

class dhgat(nn.Module):
    def __init__(self, dim):
        super(dhgat, self).__init__()
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
        
        self.discriminator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        self.src_1 = GATConv(32, 4, heads=4, dropout=0.6)
        self.pred_src = nn.Linear(16, 2)
        
        self.trg_1 = GATConv(32, 4, heads=4, dropout=0.6)
        self.pred_trg = nn.Linear(16, 2)
        
    def forward(self, src, src_edge, trg, trg_edge, device):
        src_edge = torch.tensor(src_edge[0]).to(device)
        trg_edge = torch.tensor(trg_edge[0]).to(device)
        
        src_shr = self.shr_fe(src)
        src = self.src_fe(src) + src_shr
        
        trg_shr = self.shr_fe(trg)
        trg = self.trg_fe(trg) + trg_shr
        
        # Discriminator
        ############################################################################
        s_dis_label = torch.zeros(len(src)).to(device)
        t_dis_label = torch.ones(len(trg)).to(device)
        
        src_shr_dis = GradientReversalFunction.apply(src_shr)
        src_shr_dis = self.discriminator(src_shr_dis).squeeze(1)
        s_shr_dis_loss = F.binary_cross_entropy_with_logits(src_shr_dis, s_dis_label)
        
        trg_shr_dis = GradientReversalFunction.apply(trg_shr)
        trg_shr_dis = self.discriminator(trg_shr_dis).squeeze(1)
        t_shr_dis_loss = F.binary_cross_entropy_with_logits(trg_shr_dis, t_dis_label)
        
        src_spe_dis = self.discriminator(src).squeeze(1)
        s_spe_dis_loss = F.binary_cross_entropy_with_logits(src_spe_dis, s_dis_label)
        
        trg_spe_dis = self.discriminator(trg).squeeze(1)
        t_spe_dis_loss = F.binary_cross_entropy_with_logits(trg_spe_dis, t_dis_label)
        
        disc_loss = s_shr_dis_loss + t_shr_dis_loss + s_spe_dis_loss + t_spe_dis_loss
        #############################################################################
        
        # GAT
        ####################################################
        src_x = F.relu(self.src_1(src, src_edge))
        src_out = self.pred_src(src_x)
        
        trg_x = F.relu(self.trg_1(trg, trg_edge))
        trg_out = self.pred_trg(trg_x)
        ####################################################
        
        return src_out, trg_out, disc_loss