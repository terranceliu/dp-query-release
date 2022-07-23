import os
import copy
import math
import itertools
import numpy as np
import pandas as pd
from src.utils import Dataset
from src.utils.qm import get_xy_nbin, histogramdd
from src.utils.general import get_min_dtype, add_row_convert_dtype
from src.qm import KWayMarginalQM


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