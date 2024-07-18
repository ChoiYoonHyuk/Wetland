import argparse
import json
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def tsne_plot(s_spe, s_shr, t_spe, t_shr):
    s_spe, s_shr, t_spe, t_shr = s_spe.cpu().detach().numpy(), s_shr.cpu().detach().numpy(), t_spe.cpu().detach().numpy(), t_shr.cpu().detach().numpy()
    
    label = []
    for i in range(len(s_spe)):
        label.append(0)
    for i in range(len(s_shr)):
        label.append(1)
    for i in range(len(t_shr)):
        label.append(2)
    for i in range(len(t_spe)):
        label.append(3)
    out = []
    out.extend(s_spe)
    out.extend(s_shr)
    out.extend(t_shr)
    out.extend(t_spe)
    
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(out)
    tsne_results = np.c_[tsne_results, label]
    df = pd.DataFrame(tsne_results, columns=['X', 'Y', 'class'])
    df.to_excel(excel_writer = "./tsne.xlsx")

    sns_plot = sns.scatterplot(data=df, x='X', y='Y', hue='class', palette=['green', 'orange', 'purple', 'blue'])
    handles, labels  =  sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles, ['source specific', 'source shared', 'target shared', 'target specific'], loc='upper right')
    
    da_path = './plot/tsne.png'

    sns_plot.figure.savefig(da_path)
