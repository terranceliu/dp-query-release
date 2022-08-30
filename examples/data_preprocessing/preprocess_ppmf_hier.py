import os
import json
import pickle
import argparse
import numpy as np
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

Random tracts:
56037971000 - Wyoming (56)
11001004701 - DC (11)
50007001000 - Vermont (50)
38015011101 - North Dakota (38)
02090001800 - Alaska (02)
"""

data_dir = './datasets/raw/ppmf/by_state/'
data_path_base = os.path.join(data_dir, 'ppmf_{}.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--geoid', type=str, default=None, help='tract code')
parser.add_argument('--stateid', type=str, default=None, help='selects random tract within state')
args = parser.parse_args()
assert (args.geoid is None) != (args.stateid is None)

if args.geoid is not None:
    geoid_tract = args.geoid
    state_id = geoid_tract[:2]
    ppmf_orig = pd.read_csv(data_path_base.format(state_id))
else:
    state_id = args.stateid
    ppmf_orig = pd.read_csv(data_path_base.format(state_id))

    ppmf_orig['geoid'] = ''
    ppmf_orig['geoid'] += ppmf_orig['TABBLKST'].apply(lambda x: str(x).zfill(2))
    ppmf_orig['geoid'] += ppmf_orig['TABBLKCOU'].apply(lambda x: str(x).zfill(3))
    ppmf_orig['geoid'] += ppmf_orig['TABTRACTCE'].apply(lambda x: str(x).zfill(6))
    all_geoids = ppmf_orig['geoid'].unique()
    geoid_tract = np.random.choice(all_geoids)

print(geoid_tract)

# state first
schema, ppmf = get_census_schema_and_data(ppmf_orig)

queries = build_census_queries(schema)
for q in queries:
    if 'TABBLK' in q.keys():
        del q['TABBLK']
queries = [q for q in queries if len(q) > 0]

config = {}
config['attr_cat'] = schema.column_names
config['attr_num'] = []
config['mapping_cat_domain'] = dict(zip(schema.column_names, schema.column_values))
config['mapping_num_bins'] = {}

config['attr_cat'] = [attr for attr in config['attr_cat'] if attr != 'TABBLK']
del config['mapping_cat_domain']['TABBLK']

data_config = DataPreprocessingConfig(config)
dt = DataPreprocessor(data_config)

df_preprocessed = dt.fit_transform([ppmf])
domain = dt.get_domain()

for query in queries:
    for key, values in query.items():
        encoder = dt.encoders[key]
        transformed = encoder.transform(values)
        query[key] = transformed

dataset_name = 'ppmf_hier_{}'.format(state_id)

csv_path = './datasets/{}.csv'.format(dataset_name)
df_preprocessed.to_csv(csv_path, index=False)

json_path = './datasets/domain/{}-domain.json'.format(dataset_name)
with open(json_path, 'w') as f:
    json.dump(domain, f)

queries_path = './datasets/queries/{}-set.pkl'.format(dataset_name)
with open(queries_path, 'wb') as handle:
    pickle.dump(queries, handle)

# county and tract
for i in [5, 100]:
    geoid = geoid_tract[:i]
    geolocation = GeoLocation.parse_geoid(geoid)

    ppmf = select_ppmf_geolocation(ppmf_orig, geolocation)
    _, ppmf = get_census_schema_and_data(ppmf)
    df_preprocessed = dt.transform([ppmf])

    dataset_name = 'ppmf_hier_{}'.format(geoid)

    csv_path = './datasets/{}.csv'.format(dataset_name)
    df_preprocessed.to_csv(csv_path, index=False)

    json_path = './datasets/domain/{}-domain.json'.format(dataset_name)
    with open(json_path, 'w') as f:
        json.dump(domain, f)

    queries_path = './datasets/queries/{}-set.pkl'.format(dataset_name)
    with open(queries_path, 'wb') as handle:
        pickle.dump(queries, handle)

