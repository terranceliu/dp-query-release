import os
import copy
import math
import pdb

import torch
import itertools
import numpy as np
import pandas as pd
from collections.abc import Iterable
from abc import ABC, abstractmethod

from utils.utils_data import Dataset
from utils.utils_qm import get_xy_nbin, histogramdd
from utils.utils_general import get_num_queries, get_min_dtype, add_row_convert_dtype, get_data_onehot

"""
Query manager syndata class
"""
class QueryManager(ABC):
    def __init__(self, data, workloads, sensitivity=None):
        self.domain = data.domain
        self.workloads = workloads
        self.sensitivity = sensitivity

        self.dim = np.sum(self.domain.shape)
        self.num_queries, self.workload_lens = get_num_queries(self.domain, self.workloads, return_workload_lens=True)
        self.queries = self._setup_queries()

        workload_lens = self.workload_lens.copy()
        workload_lens.insert(0, 0)
        self.workload_idxs = np.cumsum(workload_lens)
        self.workload_idxs = np.vstack([self.workload_idxs[:-1], self.workload_idxs[1:]]).T

        self.query_workload_map = np.zeros(self.num_queries, dtype=int)
        for i, (start, end) in enumerate(self.workload_idxs):
            self.query_workload_map[start:end] = i

    def regroup_answers_by_workload(self, ans):
        ans_by_workload = []
        for idxs in self.workload_idxs:
            ans_by_workload.append(ans[idxs[0]: idxs[1]])
        return ans_by_workload

    @abstractmethod
    def _setup_queries(self):
        pass

    @abstractmethod
    def get_answers(self, *args, **kwargs):
        pass

"""
Base K-way marginal query manager class
"""
class BaseKWayMarginalQM(QueryManager):
    def __init__(self, data, workloads, sensitivity=None):
        super().__init__(data, workloads, sensitivity=sensitivity)
        if sensitivity is None:
            self.sensitivity = 1 / len(data)

    def _setup_maps(self):
        self.col_map = {}
        for i, col in enumerate(self.domain.attrs):
            self.col_map[col] = i

        self.feat_pos_map = []
        cur = 0
        for sz in self.domain.shape:
            self.feat_pos_map.append(list(range(cur, cur + sz)))
            cur += sz

        self.col_pos_map = {}
        for col, i in self.col_map.items():
            self.col_pos_map[col] = self.feat_pos_map[i]

        self.pos_col_map = {}
        for i, col in enumerate(self.col_map.keys()):
            for pos in self.feat_pos_map[i]:
                attr_val = pos - self.feat_pos_map[i][0]
                self.pos_col_map[pos] = (col, attr_val)

    def _setup_queries(self):
        # Add flag variable for type of query - currently implemented with integer flags
        self._setup_maps()
        max_marginal = np.array([len(x) for x in self.workloads]).max()
        self.queries = -1 * np.ones((self.num_queries, max_marginal), dtype=get_min_dtype([self.dim]))

        idx = 0
        for feat in self.workloads:
            positions = []
            for col in feat:
                i = self.col_map[col]
                positions.append(self.feat_pos_map[i])
            x = list(itertools.product(*positions))
            x = np.array(x)
            self.queries[idx:idx + x.shape[0], :x.shape[1]] = x
            idx += x.shape[0]

        return self.queries

    def filter_query_workloads(self, workload_mask):
        self.workloads = np.array(self.workloads)[workload_mask].tolist()
        self.workload_lens = np.array(self.workload_lens)[workload_mask].tolist()

        workload_idxs = self.workload_idxs[workload_mask]
        query_mask = np.zeros(self.num_queries, dtype=bool)
        for start, end in workload_idxs:
            query_mask[start:end] = True
        self.queries = self.queries[query_mask]
        self.num_queries = len(self.queries)

        for i in range(1, len(workload_idxs)):
            end_prev = workload_idxs[i - 1, 1]
            start = workload_idxs[i, 0]
            diff = start - end_prev
            workload_idxs[i:] -= diff
        self.workload_idxs = workload_idxs

        self.query_workload_map = np.zeros(self.num_queries, dtype=int)
        for i, (start, end) in enumerate(self.workload_idxs):
            self.query_workload_map[start:end] = i

        return query_mask

"""
K-way marginal query manager
"""
class KWayMarginalQM(BaseKWayMarginalQM):
    def get_answers(self, data, weights=None, by_workload=False, density=True):
        ans_vec = []
        for proj in self.workloads:
            x = data.project(proj).datavector(weights=weights, density=density)
            ans_vec.append(x)

        if not by_workload:
            return np.concatenate(ans_vec)
        return ans_vec

    def get_query_onehot(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]

        W = []
        for q_id in q_ids:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                if p < 0:  # TODO: multiple values of k
                    break
                w[p] = 1
            W.append(w)
        W = np.array(W)
        if len(W) == 1:
            W = W.reshape(1, -1)

        return W

class KWayMarginalQMTorch(KWayMarginalQM):
    def __init__(self, data, workloads, sensitivity=None, device=None):
        super().__init__(data, workloads, sensitivity=sensitivity)
        self.device = torch.device("cpu") if device is None else device
        self.queries = torch.tensor(self.queries).long().to(self.device)

    # Currently (torch=1.11.0), torch.histogramdd doesn't support CUDA operations (rewrite below if support is added)
    def get_answers(self, data, weights=None, by_workload=False, density=True, batch_size=1000):
        if weights is None:
            weights = np.ones(len(data))
        weights = torch.tensor(weights, dtype=torch.float).unsqueeze(-1).to(self.device)

        data_onehot = torch.tensor(get_data_onehot(data)).to(self.device)
        answers = []
        for queries_batch in torch.split(self.queries, batch_size):
            answers_batch = data_onehot[:, queries_batch]
            answers_batch[:, queries_batch == -1] = True
            answers_batch = answers_batch.all(axis=-1)
            answers_batch = answers_batch * weights
            answers_batch = answers_batch.sum(0)
            answers.append(answers_batch)
        answers = torch.cat(answers)

        if density:
            answers = answers / weights.sum()
        if by_workload:
            answers = self.regroup_answers_by_workload(answers)
        return answers

"""
K-way marginal query manager
To be used with algorithms that maintain a distribution over the support (MWEM, PMW^Pub, PEP, etc.)
"""
class KWayMarginalSupportQM(KWayMarginalQM):
    def __init__(self, data, workloads, sensitivity=None,
                 cache_dir=None, overwrite_cache=True):
        super().__init__(data, workloads, sensitivity)
        self.data_support = self.get_support(data)
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache

        self.xy = None
        self.nbin = None

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.path_xy = os.path.join(cache_dir, "xy.npy")
            self.path_nbin = os.path.join(cache_dir, "nbin.npy")

        self._setup_xy_nbin()

    def get_support(self, data):
        df_support = []
        for val in list(data.domain.config.values()):
            df_support.append(np.arange(val))
        df_support = list(itertools.product(*df_support))
        df_support = np.array(df_support)
        df_support = pd.DataFrame(df_support, columns=data.df.columns)
        data_support = Dataset(df_support, data.domain)

        return data_support

    def convert_to_support_distr(self, data):
        cols = list(data.df.columns)
        new_df = data.df.groupby(cols).size().reset_index(name='Count')
        new_df = pd.merge(self.data_support.df, new_df, how='left', left_on=list(data.domain),
                          right_on=list(data.domain))
        new_df.replace(np.nan, 0)
        A_real = new_df['Count'].values
        A_real = np.nan_to_num(A_real, nan=0)
        A_real = A_real / A_real.sum()

        return A_real

    def _setup_queries(self):
        super()._setup_queries()

        domain_values = np.array(list(self.domain.config.values()))
        domain_values_cumsum = np.cumsum(domain_values)
        domain_values = domain_values.astype(get_min_dtype(domain_values))
        domain_values_cumsum = domain_values_cumsum.astype(get_min_dtype(domain_values_cumsum))

        shape = (self.num_queries, len(domain_values))
        queries = -1 * np.ones(shape, dtype=np.int8)

        idx = 0
        num_chunks = math.ceil(shape[0] / int(1e7)) # TODO: make more apaptive? right now split into chunks to avoid memory issues
        for queries_chunk in np.array_split(self.queries, num_chunks):
            x = queries_chunk[:, :, np.newaxis] - domain_values_cumsum[np.newaxis, np.newaxis, :] + domain_values
            mask = (x < domain_values) & (x >= 0)
            x[~mask] = -1
            x = x.max(axis=1)

            dtype = get_min_dtype(np.concatenate([x.flatten(), queries.flatten()])) # TODO: kind of ugly
            queries = queries.astype(dtype, copy=False)
            queries[idx:idx + x.shape[0]] = x
            idx += x.shape[0]

        # overwrite
        self.queries_onehot = self.queries
        self.queries = queries
        return self.queries

    def _load_xy_nbin(self):
        if self.cache_dir is None or self.overwrite_cache:
            return False
        if os.path.exists(self.path_xy) and os.path.exists(self.path_nbin):
            print("Loaded self.xy and self.nbin from cache.")
            self.xy = np.load(self.path_xy)
            self.nbin = np.load(self.path_nbin)
        return True

    def _setup_xy_nbin(self):
        if self._load_xy_nbin():
            return

        self.xy = None
        self.nbin = None
        for i, proj in enumerate(self.workloads):
            _data = self.data_support.project(proj)
            bins = [range(n + 1) for n in _data.domain.shape]
            xy, nbin = get_xy_nbin(_data.df.values, bins)

            if self.xy is None:
                shape = (len(self.workloads), xy.shape[0])
                self.xy = -1 * np.ones(shape, dtype=np.int8)
            if self.nbin is None:
                max_num_attr = np.max([len(x) for x in self.workloads])
                shape = (len(self.workloads), max_num_attr)
                self.nbin = -1 * np.ones(shape, dtype=np.int8)

            self.xy = add_row_convert_dtype(self.xy, xy, i)
            self.nbin = add_row_convert_dtype(self.nbin, nbin, i)

        if self.cache_dir is not None:
            np.save(self.path_xy, self.xy)
            np.save(self.path_nbin, self.nbin)

    """
    Tn these algorithms, the support is fixed and you are simply reweighting the support.
    In this case, you can predefine ``xy`` and ``nbin`` for a specific support and simply change how you weight the rows.
    This saves runtime since you can call util_qm.histogramdd instead of np.histogramdd (the latter of which needs to
    recreate ``xy`` and ``nbin`` each time it's called).
    """
    def get_answers(self, weights, by_workload=False):
        xy_neg = -1 in self.xy
        nbin_neg = -1 in self.nbin

        ans_vec = []
        for i in range(len(self.workloads)):
            xy = self.xy[i]
            nbin = self.nbin[i]
            if xy_neg:
                xy = xy[xy != -1]
            if nbin_neg:
                nbin = nbin[nbin != -1]
            x = histogramdd(xy, nbin, weights).flatten()
            ans_vec.append(x)

        if not by_workload:
            return np.concatenate(ans_vec)
        return ans_vec

    def get_support_answers(self, q_t_ind):
        query_attrs = self.queries[q_t_ind]
        query_mask = query_attrs != -1
        q_t_x = self.data_support.df.values[:, query_mask] - query_attrs[query_mask]
        q_t_x = np.abs(q_t_x).sum(axis=1)
        q_t_x = (q_t_x == 0).astype(int)
        return q_t_x

class KWayMarginalSupportQMPublic(KWayMarginalSupportQM):
    def __init__(self, data_public, workloads, sensitivity=None,
                 cache_dir=None, overwrite_cache=True):
        super().__init__(data_public, workloads, sensitivity=sensitivity,
                     cache_dir=cache_dir, overwrite_cache=overwrite_cache)
        self.histogram_public = self.convert_to_support_distr(data_public)

    def get_support(self, data_public):
        data = copy.deepcopy(data_public)
        data.df = data.df.drop_duplicates().reset_index(drop=True)
        return data