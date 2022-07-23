import numpy as np
from tqdm import tqdm

from src.qm import KWayMarginalSupportQM, KWayMarginalSupportQMPublic
from src.algo.base import IterativeAlgorithm
from src.utils.mechanisms import exponential_mech, gaussian_mech

class PEPBase(IterativeAlgorithm):
    def __init__(self, G, T, eps0,
                 alpha=0.5, max_iters=100,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(G, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.max_iters = max_iters
        self.measurements_dict = {}

    def _valid_qm(self):
        return (KWayMarginalSupportQM, KWayMarginalSupportQMPublic)

    def _project(self, q_t_A, q_t_x, m_t, offset=1e-6):
        a = np.clip(m_t, offset, 1 - offset)
        b = np.clip(q_t_A, offset, 1 - offset)
        alpha = np.log((a * (1 - b)) / ((1 - a) * b))
        factor = np.exp(q_t_x * alpha)
        self.G.A *= factor
        self.G.A /= self.G.A.sum()

    def _optimize(self):
        for _ in range(self.max_iters):
            syn_answers = self.G.get_answers(idxs=self.past_query_idxs)
            idx = np.abs(syn_answers - self.past_measurements).argmax()

            q_t_ind = self.past_query_idxs[idx]
            if q_t_ind not in self.measurements_dict.keys():
                q_t_x = self.qm.get_support_answers(q_t_ind)
                m_t = self.past_measurements[idx]
                self.measurements_dict[q_t_ind] = (q_t_x, m_t)
            else:
                q_t_x, m_t = self.measurements_dict[q_t_ind]
            q_t_A = syn_answers[idx]

            self._project(q_t_A, q_t_x, m_t)
            self.G.A_avg += self.G.A

    def fit(self, true_answers):
        if self.verbose:
            print("Fitting to query answers...")
        syn_answers = self.G.get_answers()
        scores = np.abs(true_answers - syn_answers)

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6f}".format(scores.max()))

            q_t_ind = self._sample(scores)
            m_t = self._measure(true_answers[q_t_ind])
            self._optimize()

            syn_answers = self.G.get_answers()
            scores = np.abs(true_answers - syn_answers)

        self.G.A_avg /= self.T

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