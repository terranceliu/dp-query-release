import torch
import itertools
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from abc import ABC, abstractmethod
from src.utils.general import get_num_queries, get_min_dtype, get_data_onehot

import pdb

"""
Query manager syndata class
"""
class QueryManager(ABC):
    def __init__(self, data, workloads, sensitivity=None, verbose=False):
        self.N = len(data)
        self.domain = data.domain
        self.workloads = sorted([tuple(sorted(x)) for x in workloads])
        self.sensitivity = sensitivity
        self.verbose = verbose

        self.dim = np.sum(self.domain.shape)
        self.num_workloads = len(self.workloads)
        self.num_queries, self.workload_lens = get_num_queries(self.domain, self.workloads, return_workload_lens=True)

        """
        To provide documentation, we initialize all class variables to None and (TODO) describe them in comments.
        These variables will be initialized via self._setup()
        """
        # dictionaries mapping attributes to various values
        self.col_map = None
        self.feat_pos_map = None
        self.col_pos_map = None
        self.pos_col_map = None
        # query related
        self.queries = None
        self.query_values = None
        # workload related
        self.workload_map = None
        self.workload_idxs = None
        self.query_workload_map = None

        self._setup()

    def _setup(self):
        if self.verbose:
            print("Setting up query manager...")
        self._setup_maps()
        self._setup_workloads()
        self._setup_queries()

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

    def _setup_workloads(self):
        workload_lens = self.workload_lens.copy()
        workload_lens.insert(0, 0)
        self.workload_idxs = np.cumsum(workload_lens)
        self.workload_idxs = np.vstack([self.workload_idxs[:-1], self.workload_idxs[1:]]).T

        self.workload_map = {k: v for v, k in enumerate(self.workloads)}
        self.query_workload_map = np.zeros(self.num_queries, dtype=get_min_dtype(len(self.workload_idxs)))
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
    def __init__(self, data, workloads, sensitivity=None, verbose=False):
        super().__init__(data, workloads, sensitivity=sensitivity, verbose=verbose)
        if sensitivity is None:
            self.sensitivity = 1 / self.N

    def _setup_queries(self):
        max_marginal = np.array([len(x) for x in self.workloads]).max()
        self.queries = -1 * np.ones((self.num_queries, max_marginal), dtype=get_min_dtype([self.dim]))

        domain_values = [0] + list(self.domain.config.values())
        domain_values = np.cumsum(domain_values)[:-1]
        self.query_values = self.queries.copy()

        idx = 0
        iterable = self.workloads
        if self.verbose:
            iterable = tqdm(iterable)
        for workload in iterable:
            positions = []
            for col in workload:
                i = self.col_map[col]
                positions.append(self.feat_pos_map[i])
            x = list(itertools.product(*positions))
            x = np.array(x)
            self.queries[idx:idx + x.shape[0], :x.shape[1]] = x

            dvals = domain_values[[self.col_map[c] for c in workload]]
            self.query_values[idx:idx + x.shape[0], :x.shape[1]] = x - dvals

            idx += x.shape[0]

    def filter_query_workloads(self, idxs):
        self.workloads = np.array(self.workloads)[idxs].tolist()
        self.workload_lens = np.array(self.workload_lens)[idxs].tolist()
        self.num_workloads = len(self.workloads)

        workload_idxs = self.workload_idxs[idxs]
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

    def filter_queries(self, idxs):
        self.num_queries = len(idxs)
        self.queries = self.queries[idxs]
        self.query_values = self.query_values[idxs]
        # For now, we invalidate the use of workloads when filtering out queries until there is a need to keep these variables
        self.workload_lens = None
        self.workload_idxs = None
        self.query_workload_map = None

    def get_q_desc(self, i, return_string=False):
        query = self.query_values[i]
        workload_idx = self.query_workload_map[i]
        workload = self.workloads[workload_idx]
        out = dict(zip(workload, query))
        if not return_string:
            return out

        out_str = ''
        for k, v in out.items():
            out_str += '{}={} | '.format(k, v)
        return out_str[:-3]

    def get_q_idx(self, input):
        cols = np.array(list(input.keys()))
        vals = np.array(list(input.values()))
        sort_idx = np.argsort(cols)
        cols, vals = cols[sort_idx], vals[sort_idx]

        cols = tuple(cols)
        if cols not in self.workload_map.keys():
            print("invalid set of columns")
            return None
        workload_idx = self.workload_map[cols]
        query_idxs = self.workload_idxs[workload_idx]
        q_values = self.query_values[query_idxs[0]: query_idxs[1]]
        mask = ((q_values - vals) == 0).all(-1)
        return np.arange(*query_idxs)[mask][0]

"""
K-way marginal query manager
"""
class KWayMarginalQM(BaseKWayMarginalQM):
    def get_answers(self, data, weights=None, by_workload=False, density=True):
        if self.verbose:
            print("Calculating query answers...")
        ans_vec = []
        iterable = self.workloads
        if self.verbose:
            iterable = tqdm(iterable)
        for proj in iterable:
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
    def __init__(self, data, workloads, device=None, sensitivity=None, verbose=False):
        self.device = torch.device("cpu") if device is None else device
        super().__init__(data, workloads, sensitivity=sensitivity, verbose=verbose)

    def _setup_queries(self):
        super()._setup_queries()
        self.queries = torch.tensor(self.queries).long().to(self.device)

    def get_answers_helper(self, data_onehot, weights, query_idxs=None, batch_size=1000, verbose=False):
        queries = self.queries
        if query_idxs is not None:
            queries = queries[query_idxs]
        queries_iterable = torch.split(queries, batch_size)
        if verbose:
            queries_iterable = tqdm(queries_iterable)

        answers = []
        for queries_batch in queries_iterable:
            answers_batch = data_onehot[:, queries_batch]
            # answers_batch[:, queries_batch == -1] = True
            # answers_batch = answers_batch.all(axis=-1)
            answers_batch[:, queries_batch == -1] = 1
            answers_batch = answers_batch.prod(axis=-1)
            answers_batch = answers_batch * weights
            answers_batch = answers_batch.sum(0)
            answers.append(answers_batch)
        answers = torch.cat(answers)
        return answers

    # Currently (torch=1.11.0), torch.histogramdd doesn't support CUDA operations (rewrite below if support is added)
    def get_answers(self, data, weights=None, by_workload=False, density=True, batch_size=1000):
        if self.verbose:
            print("Calculating query answers...")

        if weights is None:
            weights = np.ones(len(data))
        weights = torch.tensor(weights, dtype=torch.float).unsqueeze(-1).to(self.device)

        data_onehot = torch.tensor(get_data_onehot(data), dtype=torch.float, device=self.device)

        answers = self.get_answers_helper(data_onehot, weights, batch_size=batch_size, verbose=self.verbose)
        if density:
            answers = answers / weights.sum()
        if by_workload:
            answers = self.regroup_answers_by_workload(answers)

        return answers

class KWayMarginalAggQMTorch(KWayMarginalQMTorch):
    def __init__(self, data, workloads, sensitivity=None,
                 agg_mapping=None,
                 device=None, verbose=-False):
        super().__init__(data, workloads, sensitivity=sensitivity, device=device, verbose=verbose)

        prng = np.random.RandomState(0)
        agg_mapping = []
        for workload_idxs in self.workload_idxs:
            x = np.arange(*workload_idxs)
            for i in range(9):
                if i + 1 > len(x) - 1:
                    continue
                y = prng.choice(x, size=(i + 1), replace=False)
                agg_mapping.append(y.tolist())

        self.agg_mapping = None
        self._setup_agg_mapping(agg_mapping)

    def _setup_agg_mapping(self, agg_mapping):
        agg_mapping_k = np.array([len(x) for x in agg_mapping])
        idxs = agg_mapping_k.argsort()
        agg_mapping, agg_mapping_k = np.array(agg_mapping, dtype=object)[idxs].tolist(), agg_mapping_k[idxs]
        max_k = np.max(agg_mapping_k)

        idxs = np.unique(np.concatenate(agg_mapping))
        self.filter_queries(idxs)
        idxs_dict = {x: i for i, x in enumerate(idxs)}
        agg_mapping = [[idxs_dict[x] for x in y] for y in agg_mapping]

        agg_mapping = np.array(agg_mapping, dtype=object)
        self.agg_mapping = -np.ones((len(agg_mapping), max_k), dtype=int)
        for k in np.unique(agg_mapping_k):
            mask = agg_mapping_k == k
            self.agg_mapping[mask, :k] = np.stack(agg_mapping[mask])
        self.agg_mapping = torch.tensor(self.agg_mapping, device=self.device)

        self.num_queries = len(self.agg_mapping)

    def get_answers_helper(self, data_onehot, weights, query_idxs=None, batch_size=1000, verbose=False):
        agg_mapping = self.agg_mapping
        if query_idxs is not None:
            agg_mapping = agg_mapping[query_idxs]
        query_idxs = agg_mapping.unique()
        agg_mapping = torch.searchsorted(query_idxs, agg_mapping)

        answers = super().get_answers_helper(data_onehot, weights, query_idxs=query_idxs,
                                             batch_size=batch_size, verbose=verbose)
        answers[query_idxs == -1] = 0
        answers = answers[agg_mapping]
        answers = answers.sum(-1)
        return answers