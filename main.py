from train_test_split import data_split
from tsne import d2_plot
from preprocess import data_process
from model import plane_net
from plot import visualize
import pandas as pd

#src = pd.read_excel(r"./datasets/florida.xlsx", sheet_name="Sheet1")
src = pd.read_excel(r"./datasets/seattle.xlsx", sheet_name="Sheet1")

trg = pd.read_excel(r"./datasets/texas.xlsx", sheet_name="Sheet1")

print("Data Loaded\n")

train_mask, valid_mask, test_mask, label, split_idx, df_idx = data_split(trg)
print("Data Splitted\n")

src, trg, src_adj, trg_adj, src_label = data_process(src, trg, df_idx)
print("Data Processed\n")

#d2_plot(df, train, train_y)
x1, x2, x3 = plane_net(src, trg, src_adj, trg_adj, train_mask, valid_mask, test_mask, src_label, label, split_idx)

visualize(x1, x2, x3)