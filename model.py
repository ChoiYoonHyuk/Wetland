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


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
f1 = F1Score(task="multiclass", num_classes=5).to(device)
gaussian = False
cross_entropy_loss = False
y_mat = np.array([[1, 0.8, 0.5, 0.25, 0], [0.8, 1, 0.6, 0.3, 0], [0.25, 0.5, 1, 0.8, 0.6], [0, 0.4, 0.8, 1, 0.8], [0, 0.3, 0.6, 0.8, 1]])
y_mat = np.array([[1, 0.8, -0.1, -0.2, -0.5], [0.8, 1, -0.1, -0.2, -0.5], [-0.5, -0.2, 1, 0.5, 0.2], [-0.6, -0.2, 0.4, 1, 0.4], [-0.5, -0.2, 0.2, 0.5, 1]])


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
        

class Wetland(nn.Module):
    def __init__(self, dim):
        super(Wetland, self).__init__()
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
        
        self.fagcn = FAConv(32, eps=.5, dropout=.5)
        
        self.pred = nn.Linear(32, 2)
        
    def forward(self, src, src_edge, trg, trg_edge):
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
        
        # Heterophilic GNN
        ####################################################
        src_x = self.fagcn(src, src, src_edge)
        src_out = self.pred(self.fagcn(src_x, src, src_edge))
        
        trg_x = self.fagcn(trg, trg, trg_edge)
        trg_out = self.pred(self.fagcn(trg_x, trg, trg_edge))
        ####################################################
        
        return src_out, trg_out, disc_loss
        

class Plane(nn.Module):
    def __init__(self, dim):
        super(Plane, self).__init__()
        self.dist = nn.MSELoss()
        
        self.emb = nn.Linear(dim, 32)
        self.clf = nn.Linear(32, 5)
        
    def forward(self, x, edge):
        emb = F.dropout(F.relu(self.emb(x.to(device))), p=0.6)
        pred = self.clf(emb)
        
        return pred

def contrastive_loss(out, y):
    rare = torch.where(y < 2, 1, 0)
    rows = rare.shape[0]
    cols = 5
    rare_out = torch.zeros(rows, cols).to(device) # initializes zeros array of desired dimensions
    rare_out[list(range(rows)), rare.tolist()] = 1
    rare_out = torch.mean(out * rare_out, 1)
    
    freq = torch.where(y >= 2, 1, 0)
    rows = freq.shape[0]
    #cols = y.max() + 1
    freq_out = torch.zeros(rows, cols).to(device) # initializes zeros array of desired dimensions
    freq_out[list(range(rows)), freq.tolist()] = 1
    freq_out = torch.mean(out * freq_out, 1)
    
    loss = nn.MSELoss()
    dist = loss(rare_out, freq_out)
    
    return -dist

        
def classwise_loss(out, y):
    rows = y.shape[0]
    cols = y.max() + 1
    output = torch.zeros(rows, cols) # initializes zeros array of desired dimensions
    output[list(range(rows)), y.tolist()] = 1
    
    loss = torch.log(-out) * torch.tensor(output).to(device)

    return torch.mean(loss)
    
    
def accuracy(out, y):
    y = torch.tensor(y_mat[y.cpu(), :]).to(device)
    
    rows = out.shape[0]
    cols = 5
    output = torch.zeros(rows, cols).to(device) # initializes zeros array of desired dimensions
    output[list(range(rows)), out.tolist()] = 1
    
    result = torch.sum(y * output) / len(output)
    print(result)
    return result
    
        
def plane_net(src, trg, src_adj, trg_adj, train_mask, valid_mask, test_mask, src_label, label, split_idx):
    # Model
    print('Start Training ... \n')
    '''count = 0
    for column_name in df.columns:
        column = df[column_name]
        count += (column == 0).sum()
    print(count)'''
    # Make batch
    '''batch_size = 64
    train_ls, valid_ls, test_ls = [], [], []
    for i in train:
        train_ls.append(train[i])
    train_ls = list(chain.from_iterable(train_ls))
    train = torch.FloatTensor(df.iloc[train_ls].to_numpy())
    
    for i in valid:
        valid_ls.append(valid[i])
    valid_ls = list(chain.from_iterable(valid_ls))
    valid = torch.FloatTensor(df.iloc[valid_ls].to_numpy())
    
    for i in test:
        test_ls.append(test[i])
    test_ls = list(chain.from_iterable(test_ls))
    test = torch.FloatTensor(df.iloc[test_ls].to_numpy())'''
    
    # Gaussian augmentation
    if gaussian:
        noise = torch.zeros(len(train[0]), dtype=torch.long) + (0.05**0.5)*torch.randn(len(train[0]))
        aug_data = train + noise
        train = torch.cat((train, aug_data), 0)
        train_y.extend(train_y)
    
    res = [1 - x/sum(split_idx) for x in split_idx]
    res[1] = res[1] * 1.3
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(res).to(device))
    
    x_src = torch.FloatTensor(src.to_numpy()).to(device)
    x_trg = torch.FloatTensor(trg.to_numpy()).to(device)
        
    #tmp_train, tmp_valid, tmp_test = torch.zeros(len(train_y), 5), torch.zeros(len(valid_y), 5), torch.zeros(len(test_y), 5)
    
    #dataset = TensorDataset(Tensor(train), Tensor(train_y))
    #batch = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    #model = Plane(len(df.iloc[0]))
    lat_dim = 0
    
    if len(src.iloc[0]) > len(trg.iloc[0]):
        lat_dim = len(src.iloc[0])
        app = torch.zeros(x_trg.shape[0], len(src.iloc[0]) - len(trg.iloc[0])).to(device)
        x_trg = torch.cat((x_trg, app), 1)
    else:
        lat_dim = len(trg.iloc[0])
        app = torch.zeros(x_src.shape[0], len(trg.iloc[0]) - len(src.iloc[0])).to(device)
        x_src = torch.cat((x_src, app), 1)
    model = Wetland(lat_dim)
    
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_valid, best_test, early_stop, pred_save = 0, 0, 0, 0
    idx, x1, x2, x3 = 0, [], [], []
    stop = False
    
    for i in range(3000):
        if stop:
            break
        src_pred, trg_pred, d_loss = model(x_src, dense_to_sparse(src_adj), x_trg, dense_to_sparse(trg_adj))
        src_out, trg_out = F.log_softmax(src_pred, dim=1), F.log_softmax(trg_pred, dim=1)
        
        src_y = torch.tensor(src_label, dtype=torch.long, device=device)
        trg_y = torch.tensor(label, dtype=torch.long, device=device)
        
        if cross_entropy_loss:
            loss = criterion(out, y) + d_loss * .01
        else:
            loss = F.nll_loss(src_out, src_y) + F.nll_loss(trg_out[train_mask], trg_y[train_mask]) + d_loss * .01
        # Optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        #out = torch.where(out > 89, 1, 0)
        
        val_pred = trg_out[valid_mask]
        _, pred = val_pred.max(dim=1)
        #valid_acc = accuracy(pred, y[valid_mask])
        correct = float(pred.eq(trg_y[valid_mask]).sum().item())
        valid_acc = correct / sum(valid_mask)
        
        test_pred = trg_out[test_mask]
        _, pred = test_pred.max(dim=1)
        #test_acc = accuracy(pred, y[test_mask])
        correct = float(pred.eq(trg_y[test_mask]).sum().item())
        test_acc = correct / sum(test_mask)
        
        print(test_acc)
        #print(pred)
        
        #real_test = torch.tensor(test_y, dtype=torch.long, device=device)
        #f1_score = f1(pred, real_test)
        
        early_stop += 1
        idx += 1
        if idx % 50 == 0:
            x1.append(valid_acc)
            x2.append(test_acc)
            #x3.append(float(f1_score))
        
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
            pred_save = pred
            early_stop = 0
        
        if early_stop > 500:
            #print(sum(torch.where(pred == 0, 1, 0)) / len(pred), sum(torch.where(pred == 1, 1, 0)) / len(pred), sum(torch.where(pred == 2, 1, 0)) / len(pred), sum(torch.where(pred == 3, 1, 0)) / len(pred), sum(torch.where(pred == 4, 1, 0)) / len(pred))
            stop = True
            _, pred = test_pred.max(dim=1)
            cm = confusion_matrix(real_test.cpu(), pred.cpu())
            sns.heatmap(cm, annot=True, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig('heatmap.png')
            return x1, x2, x3
            break