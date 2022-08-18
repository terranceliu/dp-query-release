import os
import json
import pickle
import argparse
import pandas as pd

from ppmf import GeoLocation, select_ppmf_geolocation, get_census_schema_and_data, build_census_queries
from src.data_preprocessor import DataPreprocessingConfig, DataPreprocessor

import pdb

"""
Census2010_Current
https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress?form
42003140100 - CMU
42029302101 - Exton
42071011804 - Lancaster (random)
"""

parser = argparse.ArgumentParser()
parser.add_argument('--geoid', type=str)
args = parser.parse_args()
geoid = args.geoid

data_dir = './datasets/raw/ppmf/by_state/'
data_path_base = os.path.join(data_dir, 'ppmf_{}.csv')

geolocation = GeoLocation.parse_geoid(geoid)
state_id = geolocation.state_id

ppmf = pd.read_csv(data_path_base.format(state_id))
ppmf = select_ppmf_geolocation(ppmf, geolocation)
schema, ppmf = get_census_schema_and_data(ppmf)

queries = build_census_queries(schema)

config = {}
config['attr_cat'] = schema.column_names
config['attr_num'] = []
config['mapping_cat_domain'] = dict(zip(schema.column_names, schema.column_values))
config['mapping_num_bins'] = {}

data_config = DataPreprocessingConfig(config)
dt = DataPreprocessor(data_config)

df_preprocessed = dt.fit_transform([ppmf])
domain = dt.get_domain()

for query in queries:
    for key, values in query.items():
        encoder = dt.encoders[key]
        transformed = encoder.transform(values)
        query[key] = transformed

dataset_name = 'ppmf_{}'.format(geoid)
csv_path = './datasets/{}.csv'.format(dataset_name)
df_preprocessed.to_csv(csv_path, index=False)

json_path = './datasets/domain/{}-domain.json'.format(dataset_name)
with open(json_path, 'w') as f:
    json.dump(domain, f)

queries_path = './datasets/queries/{}-set.pkl'.format(dataset_name)
with open(queries_path, 'wb') as handle:
    pickle.dump(queries, handle)

