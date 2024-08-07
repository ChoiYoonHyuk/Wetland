from train_test_split import data_split
from tsne import d2_plot
from preprocess import data_process
from model import plane_net
from plot import visualize
import pandas as pd

#src = 'arizona'
src = 'texas'
#src = 'seattle'
#src = 'oregon'
#src = 'florida'
#src = 'louisiana'

trg = 'arizona'
#trg = 'texas'
#trg = 'seattle'
#trg = 'oregon'
#trg = 'florida'
#trg = 'louisiana'

wo_gnn = False

src_data = pd.read_excel("./datasets/" + src + ".xlsx", sheet_name="Sheet1")

trg_data = pd.read_excel("./datasets/" + trg + ".xlsx", sheet_name="Sheet1")

print("Data Loaded\n")

train_mask, valid_mask, test_mask, label, split_idx, df_idx = data_split(trg_data)
print("Data Splitted\n")

src, trg, src_adj, trg_adj, src_label = data_process(src_data, trg_data, df_idx, src, trg)
print("Data Processed\n")

#d2_plot(df, train, train_y)
x1, x2, x3 = plane_net(src, trg, src_adj, trg_adj, train_mask, valid_mask, test_mask, src_label, label, split_idx, wo_gnn)

visualize(x1, x2, x3)