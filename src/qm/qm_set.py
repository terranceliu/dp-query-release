import torch
import numpy as np
from tqdm import tqdm
from src.qm.qm import KWayMarginalQMTorch

class KWayMarginalSetQMTorch(KWayMarginalQMTorch):
    def __init__(self, data, queries, sensitivity=None, device=None, verbose=-False):
        workloads = self._get_workloads(queries)
        super().__init__(data, workloads, sensitivity=sensitivity, device=device, verbose=verbose)

        iter_queries = queries
        if self.verbose:
            iter_queries = tqdm(iter_queries)

        max_k = np.max([len(w) for w in workloads])
        self.queries = torch.zeros((len(queries), max_k, self.dim), dtype=bool, device=self.device)
        for query_idx, query in enumerate(iter_queries):
            for i, (attr, vals) in enumerate(query.items()):
                idxs = np.array(self.col_pos_map[attr])[vals]
                self.queries[query_idx, i, idxs] = True

        self.num_queries = len(self.queries)

    def _get_workloads(self, queries):
        workloads = [list(q.keys()) for q in queries]
        workloads = np.array(workloads, dtype=object)
        workloads = np.unique(workloads)
        workloads = list(sorted(workloads, key=len, reverse=False))
        workloads = [tuple(workload) for workload in workloads]
        return workloads

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
                x[:, ~mask.any(0)] = 1
                answers_batch[i] = x
            answers_batch = answers_batch.prod(0)
            answers_batch = answers_batch * weights
            answers_batch = answers_batch.sum(0)
            answers.append(answers_batch)
        answers = torch.cat(answers)

        return answers