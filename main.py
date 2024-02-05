from train_test_split import data_split
from preprocess import data_process
from model import plane_net
from plot import visualize
import pandas as pd

df = pd.read_excel(r"table.xlsx", sheet_name="main")

train, valid, test, train_y, valid_y, test_y = data_split(df)

df = data_process(df)

x1, x2 = plane_net(df, train, valid, test, train_y, valid_y, test_y)

visualize(x1, x2)