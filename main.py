from train_test_split import data_split
from preprocess import data_process
from model import plane_net
import pandas as pd

df = pd.read_excel(r"table.xlsx", sheet_name="main")

train, valid, test = data_split(df)

df = data_process(df)

plane_net(df, train, valid, test)