import os
import argparse
import numpy as np
import pandas as pd

import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--error_metric', type=str, default='max')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--marginal', type=int, default=None)
    parser.add_argument('--max_iters', type=int, default=None)
    args = parser.parse_args()
    return args

args = get_args()

results_path = './results/{}.csv'.format(args.filename)
df = pd.read_csv(results_path)
df.fillna('None', inplace=True)
df['rounds'] = (df['T'] * df['workload']).astype(int)

if args.dataset is not None:
    df = df[df['dataset'] == args.dataset].reset_index(drop=True)
if args.marginal is not None:
    df = df[df['marginal'] == args.marginal].reset_index(drop=True)
if args.max_iters is not None:
    df = df[df['max_iters'] == args.max_iters].reset_index(drop=True)

# datasets = df['dataset'].unique()
run_cols = ['dataset', 'marginal', 'workload', 'workload_seed', 'epsilon']
error_cols = ['error_max', 'error_mean', 'error_mean_squared']
param_cols = [col for col in df.columns if col not in ['dataset'] + run_cols + error_cols]

groupby = df.groupby(run_cols).idxmin()
# pdb.set_trace()
idx_min = groupby['error_{}'.format(args.error_metric)]

x = df.loc[idx_min, ['dataset', 'marginal', 'workload', 'epsilon', 'T', 'rounds', 'max_iters'] + error_cols]
print(x)

results_path = './results/{}-agg.csv'.format(args.filename)
x.to_csv(results_path, index=False)