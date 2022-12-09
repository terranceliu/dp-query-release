import os
import torch
import itertools
import numpy as np
import pandas as pd

def get_num_queries(domain, workloads, return_workload_lens=False):
    col_map = {}
    for i, col in enumerate(domain.attrs):
        col_map[col] = i
    feat_pos = []
    cur = 0
    for f, sz in enumerate(domain.shape):
        feat_pos.append(list(range(cur, cur + sz)))
        cur += sz

    num_queries = 0
    workload_lens = []
    for feat in workloads:
        positions = []
        for col in feat:
            i = col_map[col]
            positions.append(feat_pos[i])
        x = list(itertools.product(*positions))
        num_queries += len(x)
        workload_lens.append(len(x))

    if return_workload_lens:
        return num_queries, workload_lens
    return num_queries

def get_min_dtype(arr):
    max_val_abs = np.abs(arr).max()
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        if max_val_abs < np.iinfo(dtype).max:
            return dtype

def add_row_convert_dtype(array, row, idx):
    max_val_abs = np.abs(row).max()
    if max_val_abs > np.iinfo(array.dtype).max:
        dtype = get_min_dtype(row)
        array = array.astype(dtype)
    array[idx, :len(row)] = row
    return array

def get_data_onehot(data):
    df_data = data.df.copy()
    dim = np.sum(data.domain.shape)

    i = 0
    for attr in data.domain.attrs:
        df_data.loc[df_data[attr] >= 0, attr] += i # ignore -1
        i += data.domain[attr]
    data_values = df_data.values

    data_onehot = np.zeros((len(data_values), dim))
    arange = np.arange(len(data_values))
    arange = np.tile(arange, (data_values.shape[1], 1)).T

    assert (data_values[data_values < 0] == -1).all(), "negative values, possible overflow error due to dtype"
    x = np.tile(data_values[:, 0] + 1, (data_values.shape[-1], 1)).T
    x[data_values != -1] = 0
    data_values += x

    data_onehot[arange, data_values] = 1

    return data_onehot.astype(bool)

def get_cached_true_answers(args, data, query_manager):
    answers_dir = 'save/true_answers'
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)

    fn ='{}_{}_{}_{}.pt'.format(args.dataset, args.workload, args.marginal, args.workload_seed)
    answers_path = os.path.join(answers_dir, fn)
    if os.path.exists(answers_path):
        true_answers = torch.load(answers_path)
    else:
        true_answers = query_manager.get_answers(data)
        torch.save(true_answers, answers_path)

    return true_answers

def get_errors(true_answers, fake_answers):
    if torch.is_tensor(true_answers):
        errors = (true_answers - fake_answers).abs()
        results = {'error_max': errors.max().item(),
                   'error_mean': errors.mean().item(),
                   'error_mean_squared': torch.linalg.norm(errors, ord=2).item() ** 2 / len(errors),
                   }
    else:
        errors = np.abs(true_answers - fake_answers)
        results = {'error_max': np.max(errors),
                   'error_mean': np.mean(errors),
                   'error_mean_squared': np.linalg.norm(errors, ord=2) ** 2 / len(errors),
                   }
    return results

def get_df_results(args, errors):
    results_dict = vars(args)
    results_dict.update(errors)
    df_results = pd.Series(results_dict).to_frame().T
    return df_results

def append_results(df_results, filename, directory, remove_existing=False):
    if not os.path.exists(directory):
        os.makedirs(directory)

    results_path = os.path.join(directory, filename)
    if os.path.exists(results_path):
        if os.path.exists(results_path):
            if remove_existing:
                os.remove(results_path)
            else:
                df_existing_results = pd.read_csv(results_path)
                df_results = pd.concat((df_existing_results, df_results))

    df_results.to_csv(results_path, index=False)

def save_results(filename, directory, args, errors):
    df_results = get_df_results(args, errors)
    append_results(df_results, filename, directory)

def check_existing_results(results_fn, args):
    results_path = os.path.join('./results', results_fn)
    if not os.path.exists(results_path):
        return False
    df_results = pd.read_csv(results_path)
    for key, val in vars(args).items():
        if val is None:
            mask = df_results[key].isna()
        else:
            mask = df_results[key] == val
        df_results = df_results[mask]
    assert len(df_results) <= 1, 'duplicate results'
    return len(df_results) > 0