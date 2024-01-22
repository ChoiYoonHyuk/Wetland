import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.utils.data import DataLoader


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class Plane(nn.Module):
    def __init__(self, dim):
        super(Plane, self).__init__()
        # Loss for siamese encoder
        self.dist = nn.MSELoss()
        
        self.emb = nn.Embedding(dim, 32)

        self.clf = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Source Domain
        # Euclidean Embedding
        emb = self.emb(x.to(device))
        pred = self.clf(emb)
        print(pred)
        exit()

        return pred
        
def plane_net(df, train, valid, test):
    # Model
    print('Start Training ... \n')
    
    # Make batch
    batch_size = 32
    train_ls = []
    for i in train:
        train_ls.append(train[i])
    train_ls = list(chain.from_iterable(train_ls))
    
    batch = DataLoader(df.iloc[train_ls], batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = Plane(len(df[0]))
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for x in tqdm(batch, leave=False, total=batch_size):
        # Pre processing
        if len(x) != batch_size:
            continue

        loss = model(x)
        
        # w.write('%.4f %.4f\n' % (c_domain_loss, domain_loss))

        # Optimize
        optim.zero_grad()
        loss.backward()
        optim.step()