import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
gaussian = True

class Plane(nn.Module):
    def __init__(self, dim):
        super(Plane, self).__init__()
        # Loss for siamese encoder
        self.dist = nn.MSELoss()
        
        self.emb = nn.Linear(dim, 32)
        self.clf = nn.Linear(32, 5)
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Source Domain
        # Euclidean Embedding
        emb = F.dropout(F.relu(self.emb(x.to(device))), p=0.6)
        pred = self.clf(emb)
        
        return pred
        
def plane_net(df, train, valid, test, train_y, valid_y, test_y):
    # Model
    print('Start Training ... \n')
    
    # Make batch
    batch_size = 32
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
    
    dataset = TensorDataset(Tensor(train), Tensor(train_y))
    batch = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = Plane(len(df.iloc[0]))
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_valid, best_test, early_stop = 0, 0, 0
    idx, x1, x2 = 0, [], []
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
            loss = F.nll_loss(out, y)
            # Optimize
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            val_pred = model(valid)
            _, pred = val_pred.max(dim=1)
            y = torch.tensor(valid_y, dtype=torch.long, device=device)
            correct = float(pred.eq(y).sum().item())
            valid_acc = correct / len(valid)
            
            test_pred = model(test)
            _, pred = test_pred.max(dim=1)
            y = torch.tensor(test_y, dtype=torch.long, device=device)
            correct = float(pred.eq(y).sum().item())
            test_acc = correct / len(test)
            
            early_stop += 1
            idx += 1
            if idx % 50 == 0:
                x1.append(valid_acc)
                x2.append(test_acc)
            
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_test = test_acc
                early_stop = 0
            
            if early_stop > 100:
                print(best_test)
                stop = True
                return x1, x2
                break