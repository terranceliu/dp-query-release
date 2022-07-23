import torch
import itertools
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from abc import ABC, abstractmethod
from src.utils.general import get_num_queries, get_min_dtype, get_data_onehot

"""
Query manager syndata class
"""
class QueryManager(ABC):
    def __init__(self, data, workloads, sensitivity=None):
        self.domain = data.domain
        self.workloads = sorted([tuple(sorted(x)) for x in workloads])
        self.sensitivity = sensitivity

        self.dim = np.sum(self.domain.shape)
        self.num_queries, self.workload_lens = get_num_queries(self.domain, self.workloads, return_workload_lens=True)
        self.queries = self._setup_queries()

        workload_lens = self.workload_lens.copy()
        workload_lens.insert(0, 0)
        self.workload_idxs = np.cumsum(workload_lens)
        self.workload_idxs = np.vstack([self.workload_idxs[:-1], self.workload_idxs[1:]]).T

        self.query_workload_map = np.zeros(self.num_queries, dtype=get_min_dtype(len(self.workload_idxs)))
        for i, (start, end) in enumerate(self.workload_idxs):
            self.query_workload_map[start:end] = i
        self.workload_map = {k: v for v, k in enumerate(self.workloads)}

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

        domain_values = [0] + list(self.domain.config.values())
        domain_values = np.cumsum(domain_values)[:-1]
        self.q_values = self.queries.copy()

        idx = 0
        for workload in tqdm(self.workloads):
            positions = []
            for col in workload:
                i = self.col_map[col]
                positions.append(self.feat_pos_map[i])
            x = list(itertools.product(*positions))
            x = np.array(x)
            self.queries[idx:idx + x.shape[0], :x.shape[1]] = x

            dvals = domain_values[[self.col_map[c] for c in workload]]
            self.q_values[idx:idx + x.shape[0], :x.shape[1]] = x - dvals

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

    def get_q_desc(self, i, return_string=False):
        query = self.q_values[i]
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
        q_values = self.q_values[query_idxs[0]: query_idxs[1]]
        mask = ((q_values - vals) == 0).all(-1)
        return np.arange(*query_idxs)[mask][0]

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