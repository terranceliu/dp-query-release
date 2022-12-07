import os
import json
import pickle
import shutil
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSTravelTime, ACSMobility
from src.data_preprocessor import DataPreprocessingConfig, DataPreprocessor

import pdb

RAW_DATA_DIR = './datasets/raw/folktables'
YEAR = 2018
HORIZON = '1-Year'

ALL_STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

ALL_CONT_ATTRS = ['AGEP', 'PINCP', 'WKHP', 'PWGTP', 'INSP', 'MHP', 'MRGP', 'RMSP', 'RNTP', 'SMP', 'VALP',
                    'WATP', 'FINCP', 'GRNTP', 'GRPIP', 'HINCP', 'NOC', 'NPF', 'NRC', 'OCPIP', 'SMOCP', 'TAXAMT',
                    'WGTP1', 'INTP', 'JWMNP', 'JWRIP', 'MARHYP', 'OIP', 'PAP', 'RETP', 'SEMP', 'SSIP', 'SSP',
                    'WAGP', 'WKHP', 'YOEP', 'PERNP', 'PINCP', 'POVPIP']

COLS_DEL = ['ST']
COLS_STATE_SPECIFIC = ['PUMA', 'POWPUMA']

NUM_BINS = 10

def split_con_cat(all_attrs):
    cat = set(all_attrs) - set(ALL_CONT_ATTRS)
    con = set(all_attrs).intersection(ALL_CONT_ATTRS)
    return list(cat), list(con)

ACSTask = {
    'employment': ACSEmployment,
    'income': ACSIncome,
    'coverage': ACSPublicCoverage,
    'travel': ACSTravelTime,
    'mobility': ACSMobility,
}

def get_acs_raw(task, state, year='2018', remove_raw_files=False, return_attrs=False):
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person',
                                root_dir=RAW_DATA_DIR)
    acs_data = data_source.get_data(states=[state], download=True)
    features, target, group = ACSTask[task].df_to_numpy(acs_data)

    all_attrs = ACSTask[task].features.copy()
    df = pd.DataFrame(features, columns=all_attrs)

    target_attr = ACSTask[task].target
    all_attrs.append(target_attr)
    df[target_attr] = target.astype(features.dtype)

    if remove_raw_files:
        shutil.rmtree(data_source._root_dir)

    if return_attrs:
        attr_cat, attr_num = split_con_cat(all_attrs)
        return df, (attr_cat, attr_num)
    return df

def get_preprocessor_mappings(task, num_bins=NUM_BINS):
    dict_cat, dict_num = {}, {}
    for state in tqdm(ALL_STATES):
        df, (attr_cat, attr_num) = get_acs_raw(task, state, return_attrs=True)

        for attr in attr_cat:
            unique_attrs = set(df[attr].unique().astype(int))
            if attr in dict_cat.keys():
                dict_cat[attr] = dict_cat[attr].union(unique_attrs)
            else:
                dict_cat[attr] = unique_attrs

        for attr in attr_num:
            min_val, max_val = df[attr].min(), df[attr].max()
            if attr in dict_num.keys():
                curr_min_val, curr_max_val = dict_num[attr]
                if min_val < curr_min_val:
                    dict_num[attr][0] = min_val
                if max_val > curr_max_val:
                    dict_num[attr][1] = max_val
            else:
                dict_num[attr] = [min_val, max_val]

    for key, val in dict_cat.items():
        dict_cat[key] = list(val)
    for key, val in dict_num.items():
        dict_num[key] = list(np.linspace(*val, num=num_bins + 1))
        dict_num[key][0] = -np.inf
        dict_num[key][-1] = np.inf

    return dict_cat, dict_num

def preprocess_acs(task, state):
    df = get_acs_raw(task, state)

    mappings_dir = os.path.join(RAW_DATA_DIR, str(YEAR), HORIZON, 'preprocessor_mappings')
    if not os.path.exists(mappings_dir):
        os.makedirs(mappings_dir)
    mappings_path = os.path.join(mappings_dir, '{}.pkl'.format(task))
    if os.path.exists(mappings_path):
        with open(mappings_path, 'rb') as handle:
            dict_cat, dict_num = pickle.load(handle)
    else:
        dict_cat, dict_num = get_preprocessor_mappings(task)
        with open(mappings_path, 'wb') as handle:
            pickle.dump((dict_cat, dict_num), handle)

    for attr in COLS_DEL:
        dict_cat.pop(attr, None)
        dict_num.pop(attr, None)
    for attr in COLS_STATE_SPECIFIC:
        if attr in dict_cat.keys():
            dict_cat[attr] = list(np.unique(df[attr].unique()))
        elif attr in dict_num.keys():
            num_bins = len(list(dict_num.values())[0]) - 1
            min_val, max_val = df[attr].min(), df[attr].max()
            bins = list(np.linspace(min_val, max_val, num=num_bins + 1))
            dict_num['mapping_cat_domain'][attr] = bins

    config = {}
    config['attr_cat'] = list(dict_cat.keys())
    config['attr_num'] = list(dict_num.keys())
    config['mapping_cat_domain'] = dict_cat.copy()
    config['mapping_num_bins'] = dict_num.copy()

    data_config = DataPreprocessingConfig(config)
    dt = DataPreprocessor(data_config)

    df_preprocessed = dt.fit_transform([df])
    domain = dt.get_domain()

    # verify df and domain are correctly mapped
    for key, val in dict_cat.items():
        assert domain[key] == len(val), '{}, {}, {}'.format(key, domain[key], len(val))
        assert (df_preprocessed[key].unique() < domain[key]).all(), \
            '{}, {}, {}, {}'.format(key, df_preprocessed[key].unique(), domain[key], val)
    for key, val in dict_num.items():
        assert domain[key] == len(val) - 1, '{}, {}, {}'.format(key, domain[key], len(val) - 1)
        assert (df_preprocessed[key].unique() < domain[key]).all(), \
            '{}, {}, {}, {}'.format(key, df_preprocessed[key].unique(), domain[key], val)

    csv_path = './datasets/folktables_{}_{}_{}.csv'.format(task, YEAR, state)
    df_preprocessed.to_csv(csv_path, index=False)

    json_path = './datasets/domain/folktables_{}_{}_{}-domain.json'.format(task, YEAR, state)
    with open(json_path, 'w') as f:
        json.dump(domain, f)

    # categorical-only
    csv_path_cat = './datasets/folktables-cat_{}_{}_{}.csv'.format(task, YEAR, state)
    os.symlink(os.path.realpath(csv_path), csv_path_cat)

    for attr in config['attr_num']:
        del domain[attr]
    json_path_cat = './datasets/domain/folktables-cat_{}_{}_{}-domain.json'.format(task, YEAR, state)
    with open(json_path_cat, 'w') as f:
        json.dump(domain, f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--states', nargs='+')
    return parser.parse_args()

"""
python examples/data_preprocessing/preprocess_folktables.py \
--tasks income travel coverage mobility employment \
--states CA TX FL NY PA
"""

if __name__ == '__main__':
    args = get_args()
    acs_config_data = {'tasks': args.tasks,
                       'states': args.states,
                       }

    for task, state in itertools.product(acs_config_data['tasks'], acs_config_data['states']):
        print(task, state)
        preprocess_acs(task, state)

    shutil.rmtree(RAW_DATA_DIR)