import torch
import numpy as np

from utils.mechanisms import exponential_mech, gaussian_mech
from algorithms.gem import IterativeAlgoGEM as IterativeAlgoGEMBase
from algorithms.rap_softmax import IterativeAlgoRAPSoftmax as IterativeAlgoRAPSoftmaxBase

class IterativeAlgoGEM(IterativeAlgoGEMBase):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        scores = list(scores[x[0]:x[1]] for x in list(self.qm.workload_idxs))
        scores = np.array([x.max() for x in scores])
        max_workload_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        max_query_idxs = np.arange(*self.qm.workload_idxs[max_workload_idx])

        self.past_workload_idxs = torch.cat([self.past_workload_idxs, torch.tensor([max_workload_idx])])
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor(max_query_idxs)])
        return max_query_idxs

    def _measure(self, answers):
        noisy_answers = gaussian_mech(answers, (1 - self.alpha) * self.eps0, 2 ** 0.5 * self.qm.sensitivity)
        noisy_answers = np.clip(noisy_answers, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor(noisy_answers)])

class IterativeAlgoRAPSoftmax(IterativeAlgoRAPSoftmaxBase):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        scores = list(scores[x[0]:x[1]] for x in list(self.qm.workload_idxs))
        scores = np.array([x.max() for x in scores])
        max_workload_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        max_query_idxs = np.arange(*self.qm.workload_idxs[max_workload_idx])

        self.past_workload_idxs = torch.cat([self.past_workload_idxs, torch.tensor([max_workload_idx])])
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor(max_query_idxs)])
        return max_query_idxs

    def _measure(self, answers):
        noisy_answers = gaussian_mech(answers, (1 - self.alpha) * self.eps0, 2 ** 0.5 * self.qm.sensitivity)
        noisy_answers = np.clip(noisy_answers, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor(noisy_answers)])