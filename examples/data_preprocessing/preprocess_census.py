import json
from src.data_preprocessor import *

data_name = 'census'

config = get_config(data_name)
df_train = pd.read_csv('datasets/raw/census/train.csv')
df_test = pd.read_csv('datasets/raw/census/test.csv')

dt = DataPreprocessor(config)
dt.fit([df_train, df_test])
domain = dt.get_domain()

df_preprocessed_train = dt.transform(df_train)
df_preprocessed_test = dt.transform(df_test)
df_preprocessed = pd.concat([df_preprocessed_train, df_preprocessed_test]).reset_index(drop=True)

csv_path = "datasets/{}.csv".format(data_name)
domain_path = "datasets/domain/{}-domain.json".format(data_name)

df_preprocessed.to_csv(csv_path, index=False)
with open(domain_path, 'w') as f:
    json.dump(domain, f)