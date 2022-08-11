import torch
import itertools
import numpy as np
from tqdm import tqdm
from src.qm.qm import KWayMarginalQMTorch

import pdb

import pdb

class KWayMarginalSetQMTorch(KWayMarginalQMTorch):
    def __init__(self, data, queries, sensitivity=None, device=None, verbose=-False):
        workloads = self._get_workloads(queries)
        super().__init__(data, workloads, sensitivity=sensitivity, device=device, verbose=verbose)

        max_k = np.max([len(w) for w in workloads])
        self.queries = torch.zeros((len(queries), max_k, self.dim), dtype=bool, device=self.device)

        iter_queries = queries
        if self.verbose:
            iter_queries = tqdm(iter_queries)
        for query_idx, query in enumerate(iter_queries):
            for i, (attr, vals) in enumerate(query.items()):
                idxs = np.array(self.col_pos_map[attr])[vals]
                self.queries[query_idx, i, idxs] = True
            self.queries[query_idx, i + 1:, self.feat_pos_map[0]] = True # set idx for entire first workload to be True

        self.num_queries = len(self.queries)

        # pdb.set_trace()
        #
        # prng = np.random.RandomState(0)
        # agg_mapping = []
        # for workload_idxs in self.workload_idxs:
        #     x = np.arange(*workload_idxs)
        #     for i in range(1, 9):
        #         if i + 1 > len(x) - 1:
        #             continue
        #         y = prng.choice(x, size=(i + 1), replace=False)
        #         agg_mapping.append(y.tolist())
        #
        # self.agg_mapping = None
        # self._setup_agg_mapping(agg_mapping)

    def _get_workloads(self, queries):
        workloads = [list(q.keys()) for q in queries]
        workloads = np.array(workloads, dtype=object)
        workloads = np.unique(workloads)
        workloads = list(sorted(workloads, key=len, reverse=False))
        workloads = [tuple(workload) for workload in workloads]
        return workloads


    # def _setup_agg_mapping(self, agg_mapping):
    #     agg_mapping_k = np.array([len(x) for x in agg_mapping])
    #     idxs = agg_mapping_k.argsort()
    #     agg_mapping, agg_mapping_k = np.array(agg_mapping, dtype=object)[idxs].tolist(), agg_mapping_k[idxs]
    #     max_k = np.max(agg_mapping_k)
    #
    #     idxs = np.unique(np.concatenate(agg_mapping))
    #     self.filter_queries(idxs)
    #     idxs_dict = {x: i for i, x in enumerate(idxs)}
    #     agg_mapping = [[idxs_dict[x] for x in y] for y in agg_mapping]
    #
    #     queries_agg = [[(lambda x: np.unique(x))(y) for y in self.queries[x].T.tolist()] for x in agg_mapping]
    #     queries_agg = np.array(queries_agg, dtype=object)
    #     queries = np.zeros((queries_agg.shape[0], queries_agg.shape[1], self.dim + 1), dtype=bool)
    #     for i in range(queries_agg.shape[-1]):
    #         x = queries_agg[:, i]
    #         padded = np.array(list(zip(*itertools.zip_longest(*x, fillvalue=self.dim))))
    #         queries[np.expand_dims(np.arange(len(queries)), -1), i, padded] = True
    #     queries = queries[:, :, :-1]
    #     self.queries = torch.tensor(queries, device=self.device)
    #     self.num_queries = len(self.queries)

    def get_answers_helper(self, data_onehot, weights, query_idxs=None, batch_size=1000, verbose=False):
        queries = self.queries
        if query_idxs is not None:
            queries = queries[query_idxs]

        queries_iterable = torch.split(queries, batch_size)
        if verbose:
            queries_iterable = tqdm(queries_iterable)

        answers = []
        for queries_batch in queries_iterable:
            queries_batch = queries_batch.permute((2, 1, 0))
            answers_batch = torch.zeros((queries_batch.shape[1], len(data_onehot), queries_batch.shape[2]), device=self.device)
            for i in range(queries_batch.shape[1]):
                mask = queries_batch[:, i]
                x = data_onehot.mm(mask.float())
                answers_batch[i] = x
            answers_batch = answers_batch.prod(0)
            answers_batch = answers_batch * weights
            answers_batch = answers_batch.sum(0)
            answers.append(answers_batch)
        answers = torch.cat(answers)
        return answers