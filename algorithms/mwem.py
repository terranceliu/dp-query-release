import torch
import numpy as np

from tqdm import tqdm
from qm import KWayMarginalSupportQM
from algorithms.algo import IterativeAlgorithm, IterativeAlgorithmTorch
from utils.utils_general import get_data_onehot
from utils.mechanisms import exponential_mech, gaussian_mech

import pdb

class ApproxDistr():
    def __init__(self, qm, device=None, query_bs=1000):
        self.qm = qm
        self.device = torch.device("cpu") if device is None else device
        self.query_bs = query_bs

        self.data_support = self.qm.data_support

        self._initialize_A()

    def _initialize_A(self):
        A_init = np.ones(len(self.data_support))
        A_init /= len(A_init)
        self.A = A_init
        self.A_avg = self.A.copy()

    def get_answers(self, idxs=None, use_avg=False):
        A = self.A_avg if use_avg else self.A
        answers = self.qm.get_answers(A)
        if idxs is not None:
            return answers[idxs]
        return answers

    def get_syndata(self, num_samples=100000, use_avg=False):
        A = self.A_avg if use_avg else self.A
        return self.A

class MWEMBase(IterativeAlgorithm):
    """
        Constructors that initializes parameters
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
    def __init__(self, D, qm, T, eps0,
                 alpha=0.5, default_dir=None,
                 recycle_queries=False,
                 verbose=False, seed=None):

        super().__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.D = D
        self.recycle_queries = recycle_queries

        self.measurements_dict = {}

    def _valid_qm(self):
        return (KWayMarginalSupportQM)

    def _multiplicative_weights(self, syn_answers, q_t_ind):
        q_t_A = syn_answers[q_t_ind]
        q_t_x, m_t = self.measurements_dict[q_t_ind]
        factor = np.exp(q_t_x * (m_t - q_t_A) / 2)
        self.D.A *= factor
        self.D.A /= self.D.A.sum()

    def _optimize(self, syn_answers):
        q_t_ind = self.past_query_idxs[-1]
        m_t = self.past_measurements[-1]
        q_t_x = self.qm.get_support_answers(q_t_ind)
        self.measurements_dict[q_t_ind] = (q_t_x, m_t)

        idx = np.abs(syn_answers[self.past_query_idxs] - self.past_measurements).argmax()
        q_t_ind = self.past_query_idxs[idx]
        self._multiplicative_weights(syn_answers, q_t_ind)

        if self.recycle_queries:
            errors = np.zeros_like(self.past_query_idxs).astype(float)
            for i, idx in enumerate(self.past_query_idxs):
                q_t_A = syn_answers[idx]
                m_t = self.past_measurements[i]
                errors[i] = np.abs(q_t_A - m_t)
            mask = errors >= 0.5 * errors.max()
            update_idxs = np.array(self.past_query_idxs)[mask]
            for q_t_ind in update_idxs:
                self._multiplicative_weights(syn_answers, q_t_ind)

        self.D.A_avg += self.D.A

    """
    Algorithm fits to a list of answers.
    Input:
        true_answers (np.array): numpy array of answers the algorithm is fitting to
    """
    def fit(self, true_answers):
        syn_answers = self.D.get_answers()
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

            syn_answers = self.D.get_answers()
            scores = np.abs(true_answers - syn_answers)

        self.D.A_avg /= self.T

    def get_syndata(self, num_samples=100000, use_avg=False):
        return self.D.get_syndata(num_samples=num_samples, use_avg=use_avg)

class MWEM(MWEMBase):
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
