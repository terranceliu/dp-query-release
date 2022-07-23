import numpy as np
from tqdm import tqdm
from abc import abstractmethod

from src.qm import KWayMarginalSupportQM, KWayMarginalSupportQMPublic
from src.algo.base import IterativeAlgorithm

class IterAlgoNondpBase(IterativeAlgorithm):
    def __init__(self, G, T,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(G, T, eps0=0, alpha=0, default_dir=default_dir, verbose=verbose, seed=seed)
        self.measurements_dict = {}

    def _valid_qm(self):
        return (KWayMarginalSupportQM, KWayMarginalSupportQMPublic)

    def _sample(self, scores):
        pass

    def _measure(self, answers):
        pass

    @abstractmethod
    def _project(self, q_t_A, q_t_x, m_t):
        pass

    def fit(self, true_answers):
        if self.verbose:
            print("Fitting to query answers...")
        syn_answers = self.G.get_answers()
        scores = np.abs(true_answers - syn_answers)

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6f}".format(scores.max()))

            q_t_ind = scores.argmax()
            if q_t_ind not in self.measurements_dict.keys():
                q_t_x = self.qm.get_support_answers(q_t_ind)
                m_t = true_answers[q_t_ind]
                self.measurements_dict[q_t_ind] = (q_t_x, m_t)
            else:
                q_t_x, m_t = self.measurements_dict[q_t_ind]
            q_t_A = syn_answers[q_t_ind]

            self._project(q_t_A, q_t_x, m_t)

            syn_answers = self.G.get_answers()
            scores = np.abs(true_answers - syn_answers)

        self.G.A_avg /= self.T

class MultiplicativeWeights(IterAlgoNondpBase):
    def _project(self, q_t_A, q_t_x, m_t):
        factor = np.exp(q_t_x * (m_t - q_t_A) / 2)
        self.G.A *= factor
        self.G.A /= self.G.A.sum()

class EntropyProjection(IterAlgoNondpBase):
    def _project(self, q_t_A, q_t_x, m_t):
        offset=1e-6
        a = np.clip(m_t, offset, 1 - offset)
        b = np.clip(q_t_A, offset, 1 - offset)
        alpha = np.log((a * (1 - b)) / ((1 - a) * b))
        factor = np.exp(q_t_x * alpha)
        self.G.A *= factor
        self.G.A /= self.G.A.sum()