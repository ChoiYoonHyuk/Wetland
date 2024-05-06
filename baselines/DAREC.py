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
        

class darec(nn.Module):
    def __init__(self, dim):
        super(darec, self).__init__()
        self.fe = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Dropout(),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        self.pred = nn.Linear(32, 2)
        
    def forward(self, src, trg, device):
        src = self.fe(src)
        trg = self.fe(trg)
        
        # Discriminator
        ############################################################################
        s_dis_label = torch.zeros(len(src)).to(device)
        t_dis_label = torch.ones(len(trg)).to(device)
        
        src_shr_dis = GradientReversalFunction.apply(src)
        src_shr_dis = self.discriminator(src_shr_dis).squeeze(1)
        s_shr_dis_loss = F.binary_cross_entropy_with_logits(src_shr_dis, s_dis_label)
        
        trg_shr_dis = GradientReversalFunction.apply(trg)
        trg_shr_dis = self.discriminator(trg_shr_dis).squeeze(1)
        t_shr_dis_loss = F.binary_cross_entropy_with_logits(trg_shr_dis, t_dis_label)
        
        disc_loss = s_shr_dis_loss + t_shr_dis_loss
        #############################################################################
        
        src_out = self.pred(src)
        trg_out = self.pred(trg)
        
        return src_out, trg_out, disc_loss