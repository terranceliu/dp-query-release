import time
import argparse
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from qm import KWayMarginalSupportQM
from algorithms.base.algo import IterativeAlgorithm
from utils.mechanisms import exponential_mech, gaussian_mech

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
        max_query_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs.append(max_query_idx)
        return max_query_idx

    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = np.clip(noisy_answer, 0, 1)
        self.past_measurements.append(noisy_answer)
        return noisy_answer

    """
    Algorithm fits to a list of answers.
    Input:
        true_answers (np.array): numpy array of answers the algorithm is fitting to
    """
    def fit(self, true_answers):
        self.A_init = self._initialize_A()
        A = np.copy(self.A_init)
        A_avg = np.zeros(self.A_init.shape)

        self.measurements_dict = {}

        fake_answers = self.qm.get_answers(A)
        scores = np.abs(true_answers - fake_answers)
        pbar = tqdm(range(self.T))
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6f}".format(scores.max()))

            # SAMPLE
            q_t_ind = self._sample(scores)

            # MEASURE
            m_t = self._measure(true_answers[q_t_ind])

            # Multiplicative Weights update
            query_attrs = self.qm.queries[q_t_ind]
            query_mask = query_attrs != -1
            q_t_x = self.data_support.df.values[:, query_mask] - query_attrs[query_mask]
            q_t_x = np.abs(q_t_x).sum(axis=1)
            q_t_x = (q_t_x == 0).astype(int)

            self.measurements_dict[q_t_ind] = (q_t_x, m_t)

            mw_update_queries = [q_t_ind]
            if self.recycle_queries:
                errors_dict = {}
                for idx, (q_t_x, m_t) in self.measurements_dict.items():
                    q_t_A = fake_answers[idx]
                    errors_dict[idx] = np.abs(m_t - q_t_A).max()
                mask = np.array(list(errors_dict.values())) >= 0.5 * errors_dict[q_t_ind]
                past_indices = np.array(list(errors_dict.keys()))[mask]
                np.random.shuffle(past_indices)
                mw_update_queries += list(past_indices)

            for i in mw_update_queries:
                q_t_A = fake_answers[i]
                q_t_x, m_t = self.measurements_dict[i]
                factor = np.exp(q_t_x * (m_t - q_t_A) / 2)
                A = A * factor
                A = A / A.sum()

            A_avg += A

            fake_answers = self.qm.get_answers(A)
            scores = np.abs(true_answers - fake_answers)

        A_last = np.copy(A)
        A_avg /= self.T

        self.A_last = A_last
        self.A_avg = A_avg

    """
    Returns synthetic data in some form
    """
    def get_syndata(self, return_avg=False):
        if return_avg:
            return self.A_avg
        return self.A_last

def get_args():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, default='adult-reduced')
    parser.add_argument('--marginal', type=int, default=3)
    parser.add_argument('--workload', type=int, default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    # privacy args
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=1.0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    # general algo args
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test_seed', type=int, default=None)

    # MWEM specific params
    parser.add_argument('--recycle', action='store_true', help='reuse past queries for MW')

    args = parser.parse_args()

    print(args)
    return args