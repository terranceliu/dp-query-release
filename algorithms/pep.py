import torch
import numpy as np

from algorithms.mwem import MWEMBase
from utils.utils_general import get_data_onehot
from utils.mechanisms import exponential_mech, gaussian_mech

import pdb

class PEPBase(MWEMBase):
    def __init__(self, qm, T, eps0, device=None,
                 alpha=0.5, max_iters=100, query_bs=1000,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.device = torch.device("cpu") if device is None else device # CUDA recommended only for single-query accounting
        self.max_iters = max_iters
        self.query_bs = query_bs

        self.data_support_onehot = get_data_onehot(self.data_support)
        self.data_support_onehot = torch.tensor(self.data_support_onehot, device=self.device)
        self.queries_onehot = torch.tensor(self.qm.queries_onehot, device=self.device).long()

    def get_query_answers(self, idxs=None):
        if self.device.type == 'cpu':
            return self.qm.get_answers(self.A)[idxs]
        else: # This seems to only be faster when idxs is small (i.e., single query accounting)
            A = torch.tensor(self.A, device=self.device)
            queries = self.queries_onehot[idxs]
            answers = []
            for _queries in torch.split(queries, self.query_bs):
                x = self.data_support_onehot[:, _queries].all(-1)
                x = x * A.unsqueeze(-1)
                x = x.sum(0)
                answers.append(x)
            answers = torch.cat(answers)
            return answers.cpu().numpy()

    def _optimize(self, syn_answers):
        for _ in range(self.max_iters):
            syn_answers = self.get_query_answers(idxs=self.past_query_idxs)
            idx = np.abs(syn_answers - self.past_measurements).argmax()
            q_t_ind = self.past_query_idxs[idx]
            m_t = self.past_measurements[idx]
            q_t_A = syn_answers[idx]
            if q_t_ind in self.measurements_dict.keys():
                q_t_x, m_t = self.measurements_dict[q_t_ind]
            else:
                q_t_x = self._get_support_answers(q_t_ind)
                self.measurements_dict[q_t_ind] = (q_t_x, m_t)

            offset = 1e-6
            real = np.clip(m_t, offset, 1 - offset)
            fake = np.clip(q_t_A, offset, 1 - offset)
            temp = (real * (1 - fake)) / ((1 - real) * fake)
            alpha = np.log(temp)
            factor = np.exp(q_t_x * alpha)

            self.A *= factor
            self.A /= self.A.sum()

class PEP(PEPBase):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        scores = list(scores[x[0]:x[1]] for x in list(self.qm.workload_idxs))
        scores = np.array([x.max().item() for x in scores])
        max_workload_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        max_query_idxs = list(np.arange(*self.qm.workload_idxs[max_workload_idx]))

        self.past_workload_idxs.append(max_workload_idx)
        self.past_query_idxs += max_query_idxs
        return max_query_idxs

    def _measure(self, answers):
        noisy_answers = gaussian_mech(answers, (1 - self.alpha) * self.eps0, 2 ** 0.5 * self.qm.sensitivity)
        noisy_answers = list(np.clip(noisy_answers, 0, 1))
        self.past_measurements += noisy_answers
        return noisy_answers

class PEPSingle(PEPBase):
    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = np.clip(noisy_answer, 0, 1)
        self.past_measurements.append(noisy_answer)
        return noisy_answer

    def _get_support_answers(self, q_t_ind):
        query_attrs = self.qm.queries[q_t_ind]
        query_mask = query_attrs != -1
        q_t_x = self.data_support.df.values[:, query_mask] - query_attrs[query_mask]
        q_t_x = np.abs(q_t_x).sum(axis=1)
        q_t_x = (q_t_x == 0).astype(int)
        return q_t_x