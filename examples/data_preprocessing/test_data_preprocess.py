import pandas as pd

from src.data_preprocessor import get_config

import pdb

df_train = pd.read_csv('datasets/raw/census/train.csv')
df_test = pd.read_csv('datasets/raw/census/test.csv')
df = pd.concat([df_train, df_test]).reset_index(drop=True)
print(df.shape)

config = get_config('census')

pdb.set_trace()

col = 'wage_per_hour'
mask = df[col] != 0
df.loc[mask, col].describe()

for col in config.attr_cat: print(col, len(df[col].unique()))

for col in config.attr_num: print(col, len(df[col].unique()))