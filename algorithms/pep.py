import torch
import numpy as np

from utils.utils_general import get_data_onehot
from algorithms.mwem import MWEM

class PEP(MWEM):
    def __init__(self, qm, T, eps0, device=None,
                 alpha=0.5, max_iters=10,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.device = torch.device("cpu") if device is None else device
        self.max_iters = max_iters

        self.data_support_onehot = get_data_onehot(self.data_support)
        self.data_support_onehot = torch.tensor(self.data_support_onehot, device=self.device)
        self.queries_onehot = torch.tensor(self.qm.queries_onehot, device=self.device).long()

    def get_query_answers(self, idxs=None):
        if self.device.type == 'cpu':
            return self.qm.get_answers(self.A)[idxs]
        else:
            A = torch.tensor(self.A, device=self.device)
            queries = self.queries_onehot[idxs]
            answers = self.data_support_onehot[:, queries].all(-1)
            answers = answers * A.unsqueeze(-1)
            answers = answers.sum(0)
            return answers.cpu().numpy()

    def _optimize(self, syn_answers):
        q_t_ind = self.past_query_idxs[-1]
        m_t = self.past_measurements[-1]
        q_t_x = self._get_support_answers(q_t_ind)
        self.measurements_dict[q_t_ind] = (q_t_x, m_t)

        for _ in range(self.max_iters):
            syn_answers = self.get_query_answers(idxs=self.past_query_idxs)
            idx = np.abs(syn_answers - self.past_measurements).argmax()
            q_t_ind = self.past_query_idxs[idx]
            q_t_A = syn_answers[idx]
            q_t_x, m_t = self.measurements_dict[q_t_ind]

            offset = 1e-6
            real = np.clip(m_t, offset, 1 - offset)
            fake = np.clip(q_t_A, offset, 1 - offset)
            temp = (real * (1 - fake)) / ((1 - real) * fake)
            alpha = np.log(temp)
            factor = np.exp(q_t_x * alpha)

            self.A *= factor
            self.A /= self.A.sum()