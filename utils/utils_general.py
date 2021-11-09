import itertools
import numpy as np
from utils import cdp_rho

def get_per_round_budget_zCDP(epsilon, delta, T, alpha=None):
    rho = cdp_rho(epsilon, delta)
    if alpha is None:
        eps0 = 2 * rho / T
    else:
        eps0 = (2 * rho) / (T * (alpha ** 2 + (1 - alpha) ** 2))
    eps0 = eps0 ** 0.5
    return eps0, rho

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

def get_errors(true_answers, fake_answers):
    errors = np.abs(true_answers - fake_answers)
    results = {'max': np.max(errors),
               'mean': np.mean(errors),
               'mean_squared': np.linalg.norm(errors, ord=2) ** 2 / len(errors),
               }
    return results