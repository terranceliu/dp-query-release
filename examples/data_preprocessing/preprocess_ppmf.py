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

def get_dt(schema):
    config = {}
    config['attr_cat'] = schema.column_names
    config['attr_num'] = []
    config['mapping_cat_domain'] = dict(zip(schema.column_names, schema.column_values))
    config['mapping_num_bins'] = {}

    data_config = DataPreprocessingConfig(config)
    dt = DataPreprocessor(data_config)
    return dt

def get_queries(schema, dt):
    queries = build_census_queries(schema)
    for query in queries:
        for key, values in query.items():
            encoder = dt.encoders[key]
            transformed = encoder.transform(values)
            query[key] = transformed
    return queries

def save_files(dataset_name, df_preprocessed, domain, queries):
    csv_path = './datasets/ppmf/{}.csv'.format(dataset_name)
    df_preprocessed.to_csv(csv_path, index=False)

    json_path = './datasets/ppmf/domain/{}-domain.json'.format(dataset_name)
    with open(json_path, 'w') as f:
        json.dump(domain, f)

    queries_path = './datasets/ppmf/queries/{}-set.pkl'.format(dataset_name)
    with open(queries_path, 'wb') as handle:
        pickle.dump(queries, handle)



parser = argparse.ArgumentParser()
parser.add_argument('--geoid', type=str, default=None, help='tract code')
parser.add_argument('--stateid', type=str, default=None, help='selects random tract within state')
parser.add_argument('--seed', type=int, default=0, help='seed for selecting random tract')
parser.add_argument('--blocks', action='store_true', help='generates files for all blocks')
args = parser.parse_args()
assert (args.geoid is None) != (args.stateid is None)

data_dir = './datasets/raw/ppmf/by_state/'
data_path_base = os.path.join(data_dir, 'ppmf_{}.csv')

base_dir = './datasets/ppmf'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    os.makedirs(os.path.join(base_dir, 'domain'))
    os.makedirs(os.path.join(base_dir, 'queries'))

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

    prng = np.random.RandomState(args.seed)
    geoid_tract = prng.choice(all_geoids)

print(geoid_tract)

# state (remove block)
ppmf = ppmf_orig.copy()
ppmf['TABBLK'] = -1

schema, ppmf = get_census_schema_and_data(ppmf)
dt = get_dt(schema)

df_preprocessed = dt.fit_transform([ppmf])
domain = dt.get_domain()
queries = get_queries(schema, dt)

dataset_name = 'ppmf_{}'.format(state_id)
save_files(dataset_name, df_preprocessed, domain, queries)


# county (remove block)
geoid = geoid_tract[:5]
geolocation = GeoLocation.parse_geoid(geoid)
ppmf = select_ppmf_geolocation(ppmf_orig, geolocation)
ppmf['TABBLK'] = -1

schema, ppmf = get_census_schema_and_data(ppmf)
dt = get_dt(schema)

df_preprocessed = dt.fit_transform([ppmf])
domain = dt.get_domain()
queries = get_queries(schema, dt)

dataset_name = 'ppmf_{}'.format(geoid)
save_files(dataset_name, df_preprocessed, domain, queries)


# tract
geoid = geoid_tract
geolocation = GeoLocation.parse_geoid(geoid)
ppmf = select_ppmf_geolocation(ppmf_orig, geolocation)

schema, ppmf = get_census_schema_and_data(ppmf)
dt = get_dt(schema)

df_preprocessed = dt.fit_transform([ppmf])
domain = dt.get_domain()
queries = get_queries(schema, dt)

dataset_name = 'ppmf_{}'.format(geoid)
save_files(dataset_name, df_preprocessed, domain, queries)

if args.blocks:
    geolocation = GeoLocation.parse_geoid(geoid_tract)
    ppmf_tract = select_ppmf_geolocation(ppmf_orig, geolocation)
    print(ppmf_tract.groupby('TABBLK').size())
    all_blockids = ppmf_tract['TABBLK'].unique()
    all_blockids = [str(x).zfill(4) for x in all_blockids]
    for blockid in all_blockids:
        geoid = geoid_tract + blockid
        geolocation = GeoLocation.parse_geoid(geoid)
        ppmf = select_ppmf_geolocation(ppmf_orig, geolocation)

        schema, ppmf = get_census_schema_and_data(ppmf_orig)
        dt = get_dt(schema)

        df_preprocessed = dt.fit_transform([ppmf])
        domain = dt.get_domain()
        queries = get_queries(schema, dt)

        dataset_name = 'ppmf_{}'.format(state_id)
        save_files(dataset_name, df_preprocessed, domain, queries)