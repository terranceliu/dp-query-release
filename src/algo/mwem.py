import numpy as np
from tqdm import tqdm

from src.qm import KWayMarginalSupportQM, KWayMarginalSupportQMPublic
from src.algo.base import IterativeAlgorithm
from src.utils.mechanisms import exponential_mech, gaussian_mech

class MWEMBase(IterativeAlgorithm):
    """
        Constructors that initializes parameters
        Input:
        G (ApproxDistr): maintains approximating distribution A
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
    def __init__(self, G, T, eps0,
                 alpha=0.5, recycle_queries=False,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(G, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.recycle_queries = recycle_queries
        self.measurements_dict = {}

    def _valid_qm(self):
        return (KWayMarginalSupportQM, KWayMarginalSupportQMPublic)

    def _multiplicative_weights(self, syn_answers, q_t_ind):
        q_t_A = syn_answers[q_t_ind]
        q_t_x, m_t = self.measurements_dict[q_t_ind]
        factor = np.exp(q_t_x * (m_t - q_t_A) / 2)
        self.G.A *= factor
        self.G.A /= self.G.A.sum()

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
            self._optimize(syn_answers)

            syn_answers = self.G.get_answers()
            scores = np.abs(true_answers - syn_answers)

        self.G.A_avg /= self.T

# Not implemented
class MWEM(MWEMBase):
    pass

class MWEMSingle(MWEMBase):
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
