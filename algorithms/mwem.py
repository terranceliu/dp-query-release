import argparse
import numpy as np

from tqdm import tqdm
from qm import KWayMarginalSupportQM
from algorithms.algo import IterativeAlgorithm
from utils.mechanisms import exponential_mech, gaussian_mech

import pdb

class MWEM(IterativeAlgorithm):
    """
        Constructors that initializes paramters
        Input:
        qm (QueryManager): query manager for defining queries and calculating answers
        T (int): Number of rounds to run algorithm
        eps0 (float): Privacy budget per round (zCDP)
        N (int): shape of DataFrame/dataset
        data(pd.DataFrame):
        alpha (float, optional): Changes the allocation of the per-round privacy budget
            Selection mechanism uses ``alpha * eps0`` and Measurement mechanism uses ``(1-alpha) * eps0``.
            If given, it must be between 0 and 1.
        save_path (string, optional): Path for saving the class state
        recycle_queries (QueryManager?): past queries
        seed (int): seed set to reproduce results (if needed)
    """
    def __init__(self, qm, T, eps0,
                 alpha=0.5, default_dir=None,
                 recycle_queries=False,
                 verbose=False, seed=None):

        super().__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)

        self.recycle_queries = recycle_queries

        self.data_support = self.qm.data_support
        self.A_init = None
        self.A_avg = None
        self.A_last = None

        self.measurements_dict = None

    def _valid_qm(self):
        return (KWayMarginalSupportQM)

    def _initialize_A(self):
        df_support = self.data_support.df
        A_init = np.ones(len(df_support))
        A_init /= len(A_init)
        return A_init

    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        max_query_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs.append(max_query_idx)
        return max_query_idx

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

    def _optimize(self, syn_answers):
        q_t_ind = self.past_query_idxs[-1]
        m_t = self.past_measurements[-1]
        q_t_x = self._get_support_answers(q_t_ind)
        self.measurements_dict[q_t_ind] = (q_t_x, m_t)

        idx = np.abs(syn_answers[self.past_query_idxs] - self.past_measurements).argmax()
        q_t_ind = self.past_query_idxs[idx]

        mw_update_queries = [q_t_ind]
        if self.recycle_queries:
            errors_dict = {}
            for idx, (q_t_x, m_t) in self.measurements_dict.items():
                q_t_A = syn_answers[idx]
                errors_dict[idx] = np.abs(m_t - q_t_A).max()
            mask = np.array(list(errors_dict.values())) >= 0.5 * errors_dict[q_t_ind]
            past_indices = np.array(list(errors_dict.keys()))[mask]
            np.random.shuffle(past_indices)
            mw_update_queries += list(past_indices)

        for i in mw_update_queries:
            q_t_A = syn_answers[i]
            q_t_x, m_t = self.measurements_dict[i]
            factor = np.exp(q_t_x * (m_t - q_t_A) / 2)
            self.A *= factor
            self.A /= self.A.sum()

        self.A_avg += self.A

    """
    Algorithm fits to a list of answers.
    Input:
        true_answers (np.array): numpy array of answers the algorithm is fitting to
    """
    def fit(self, true_answers):
        self.A_init = self._initialize_A()
        self.A = np.copy(self.A_init)
        self.A_avg = np.zeros(self.A_init.shape)

        self.measurements_dict = {}

        syn_answers = self.qm.get_answers(self.A)
        scores = np.abs(true_answers - syn_answers)
        pbar = tqdm(range(self.T))
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6f}".format(scores.max()))

            # SAMPLE
            q_t_ind = self._sample(scores)

            # MEASURE
            m_t = self._measure(true_answers[q_t_ind])

            # Multiplicative Weights
            self._optimize(syn_answers)

            syn_answers = self.qm.get_answers(self.A)
            scores = np.abs(true_answers - syn_answers)

        self.A_avg /= self.T

    """
    Returns synthetic data in some form
    """
    def get_syndata(self, return_avg=False):
        if return_avg:
            return self.A_avg
        return self.A