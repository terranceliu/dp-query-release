import os
import argparse
import numpy as np
import pandas as pd

from ppmf import GeoLocation, select_ppmf_geolocation, get_census_schema_and_data
from preprocess_ppmf import get_dt, get_queries, save_files

import pdb

def check_paths_exist(dataset_name):
    csv_path = './datasets/ppmf/{}.csv'.format(dataset_name)
    json_path = './datasets/ppmf/domain/{}-domain.json'.format(dataset_name)
    queries_path = './datasets/ppmf/queries/{}-set.pkl'.format(dataset_name)

    for path in [csv_path, json_path, queries_path]:
        if not os.path.exists(path):
            return False
    return True

def process_geolocations(geolocations, remove_block_attr=False):
    for geolocation in geolocations:
        dataset_name = 'ppmf_{}'.format(geolocation.to_geoid())
        if check_paths_exist(dataset_name):
            continue

        ppmf = select_ppmf_geolocation(ppmf_orig, geolocation)
        if remove_block_attr:
            ppmf['TABBLK'] = -1

        schema, ppmf = get_census_schema_and_data(ppmf)
        dt = get_dt(schema)

        df_preprocessed = dt.fit_transform([ppmf])
        domain = dt.get_domain()
        queries = get_queries(schema, dt)

        save_files(dataset_name, df_preprocessed, domain, queries)

FACTORS = [1, 2, 4, 8, 16, 32, 64]

parser = argparse.ArgumentParser()
parser.add_argument('--stateid', type=str, default=None, help='selects random tract within state')
args = parser.parse_args()

data_dir = './datasets/raw/ppmf/by_state/'
data_path_base = os.path.join(data_dir, 'ppmf_{}.csv')

base_dir = './datasets/ppmf'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    os.makedirs(os.path.join(base_dir, 'domain'))
    os.makedirs(os.path.join(base_dir, 'queries'))

state_id = args.stateid
ppmf_orig = pd.read_csv(data_path_base.format(state_id))

ppmf_orig['geoid'] = ''
ppmf_orig['geoid'] += ppmf_orig['TABBLKST'].apply(lambda x: str(x).zfill(2))
ppmf_orig['geoid'] += ppmf_orig['TABBLKCOU'].apply(lambda x: str(x).zfill(3))
ppmf_orig['geoid'] += ppmf_orig['TABTRACTCE'].apply(lambda x: str(x).zfill(6))
ppmf_orig['geoid'] += ppmf_orig['TABBLK'].apply(lambda x: str(x).zfill(4))

block_sizes = ppmf_orig.groupby('geoid').size().sort_values()

target_sizes = [block_sizes.max() / i for i in FACTORS] + [block_sizes.mean()] + [block_sizes.median()]
geoids = [(block_sizes - target_size).abs().idxmin() for target_size in target_sizes]
if len(np.unique(geoids)) != len(target_sizes):
    print("Warning [stateid: {}]: duplicate blocks".format(state_id))

print("TRACTS+=({})".format(' '.join(geoids)))

# block
geolocations = [GeoLocation.parse_geoid(geoid) for geoid in geoids]
process_geolocations(geolocations, remove_block_attr=True)

# track
geolocations = [g.set_block(None) for g in geolocations]
process_geolocations(geolocations)

# county
geolocations = [g.set_census_tract(None) for g in geolocations]
process_geolocations(geolocations, remove_block_attr=True)

# state
geolocations = [g.set_county_id(None) for g in geolocations]
process_geolocations(geolocations, remove_block_attr=True)