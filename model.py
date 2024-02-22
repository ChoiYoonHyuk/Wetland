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
from sklearn.metrics import confusion_matrix


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
f1 = F1Score(task="multiclass", num_classes=5).to(device)
gaussian = False
cross_entropy_loss = False
y_mat = np.array([[1, 0.8, 0.5, 0.25, 0], [0.8, 1, 0.6, 0.3, 0], [0.25, 0.5, 1, 0.8, 0.6], [0, 0.4, 0.8, 1, 0.8], [0, 0.3, 0.6, 0.8, 1]])
y_mat = np.array([[1, 0.8, -0.1, -0.2, -0.5], [0.8, 1, -0.1, -0.2, -0.5], [-0.5, -0.2, 1, 0.5, 0.2], [-0.6, -0.2, 0.4, 1, 0.4], [-0.5, -0.2, 0.2, 0.5, 1]])

class Plane(nn.Module):
    def __init__(self, dim):
        super(Plane, self).__init__()
        self.dist = nn.MSELoss()
        
        self.emb = nn.Linear(dim, 32)
        self.clf = nn.Linear(32, 5)
        
    def forward(self, x):
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
    y = torch.tensor(y_mat[y, :]).to(device)
    
    rows = out.shape[0]
    cols = 5
    output = torch.zeros(rows, cols).to(device) # initializes zeros array of desired dimensions
    output[list(range(rows)), out.tolist()] = 1
    
    result = torch.sum(y * output) / len(output)
    print(result)
    return result
    
        
def plane_net(df, train, valid, test, train_y, valid_y, test_y, split_idx):
    # Model
    print('Start Training ... \n')
    '''count = 0
    for column_name in df.columns:
        column = df[column_name]
        count += (column == 0).sum()
    print(count)'''
    
    # Make batch
    batch_size = 64
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
    test = torch.FloatTensor(df.iloc[test_ls].to_numpy())
    
    # Gaussian augmentation
    if gaussian:
        noise = torch.zeros(len(train[0]), dtype=torch.long) + (0.05**0.5)*torch.randn(len(train[0]))
        aug_data = train + noise
        train = torch.cat((train, aug_data), 0)
        train_y.extend(train_y)
    
    res = [1 - x/sum(split_idx) for x in split_idx]
    res[1] = res[1] * 1.3
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(res).to(device))
    tmp_train, tmp_valid, tmp_test = torch.zeros(len(train_y), 5), torch.zeros(len(valid_y), 5), torch.zeros(len(test_y), 5)
    
    dataset = TensorDataset(Tensor(train), Tensor(train_y))
    batch = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = Plane(len(df.iloc[0]))
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_valid, best_test, early_stop, pred_save = 0, 0, 0, 0
    idx, x1, x2, x3 = 0, [], [], []
    stop = False
    
    for i in range(50):
        if stop:
            break
        for x in batch:
            # Pre processing
            if len(x[0]) != batch_size:
                continue
            
            pred = model(x[0])
            out = F.log_softmax(pred, dim=1)
            y = torch.tensor(x[1], dtype=torch.long, device=device)
        
            if cross_entropy_loss:
                loss = criterion(out, y)
            else:
                loss = F.nll_loss(out, y) + contrastive_loss(pred, y) * 0.1
                #loss = classwise_loss(out, y)
            # Optimize
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            val_pred = model(valid)
            _, pred = val_pred.max(dim=1)
            valid_acc = accuracy(pred, valid_y)
            '''val_y = torch.tensor(valid_y, dtype=torch.long, device=device)
            correct = float(pred.eq(val_y).sum().item())
            valid_acc = correct / len(valid)'''
            
            test_pred = model(test)
            _, pred = test_pred.max(dim=1)
            test_acc = accuracy(pred, test_y)
            real_test = torch.tensor(test_y, dtype=torch.long, device=device)
            f1_score = f1(pred, real_test)
            '''correct = float(pred.eq(test_y).sum().item())
            test_acc = correct / len(test)'''
            
            early_stop += 1
            idx += 1
            if idx % 50 == 0:
                x1.append(valid_acc)
                x2.append(test_acc)
                x3.append(float(f1_score))
            
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_test = test_acc
                pred_save = pred
                early_stop = 0
            
            if early_stop > 100:
                print(best_test, f1_score)
                print(sum(torch.where(pred == 0, 1, 0)) / len(pred), sum(torch.where(pred == 1, 1, 0)) / len(pred), sum(torch.where(pred == 2, 1, 0)) / len(pred), sum(torch.where(pred == 3, 1, 0)) / len(pred), sum(torch.where(pred == 4, 1, 0)) / len(pred))
                stop = True
                _, pred = test_pred.max(dim=1)
                cm = confusion_matrix(real_test.cpu(), pred.cpu())
                sns.heatmap(cm, annot=True, cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig('heatmap.png')
                return x1, x2, x3
                break