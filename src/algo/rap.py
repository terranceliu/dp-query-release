import torch
import numpy as np

from tqdm import tqdm
from torch import optim

from src.qm import KWayMarginalQM
from src.algo.base import IterativeAlgorithmTorch
from src.utils.mechanisms import exponential_mech, gaussian_mech

class IterAlgoRAPBase(IterativeAlgorithmTorch):
    def __init__(self, G, T, eps0,
                 alpha=0.5, default_dir=None, verbose=False, seed=None,
                 samples_per_round=1, lr=1e-4, max_iters=1000, max_idxs=10000):
        super().__init__(G, T, eps0, alpha=alpha,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        self.samples_per_round = samples_per_round

        # learning parameters
        self.lr = lr
        self.max_iters = max_iters
        self.max_idxs = max_idxs

        self.optimizer = optim.Adam(self.G.generator.parameters(), lr=self.lr)

    def _valid_qm(self):
        return (KWayMarginalQM)

    def _get_loss(self):
        idxs = None if self.max_idxs is None else torch.randperm(len(self.past_query_idxs))[:self.max_idxs]
        errors = self._get_sampled_query_errors(idxs=idxs)
        loss = torch.norm(errors, p=2) ** 2
        return loss

    def fit(self, true_answers):
        if self.verbose:
            print("Fitting to query answers...")
        syn_answers = self.G.get_qm_answers()
        scores = (true_answers - syn_answers).abs()

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6}".format(scores.max().item()))

            for _ in range(self.samples_per_round):
                max_query_idx = self._sample(scores)
                self._measure(true_answers[max_query_idx])

            # scheduler is just used for early stopping
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.max_iters,
                                                             threshold=1e-7, threshold_mode='rel')

            for step in range(self.max_iters):
                self.optimizer.zero_grad()

                loss = self._get_loss()
                if loss < 1e-8:
                    break

                loss.backward()
                scheduler.step(loss)
                if scheduler.num_bad_epochs > 10:
                    break

                self.optimizer.step()
                # self.rt.clip_weights()

            syn_answers = self.G.get_qm_answers()
            scores = (true_answers - syn_answers).abs()
            self.record_errors(true_answers, syn_answers)

class IterAlgoRAP(IterAlgoRAPBase):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        scores = list(scores[x[0]:x[1]] for x in list(self.qm.workload_idxs))
        scores = np.array([x.max().item() for x in scores])
        max_workload_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        max_query_idxs = torch.arange(*self.qm.workload_idxs[max_workload_idx], device=self.device)

        self.past_workload_idxs = torch.cat([self.past_workload_idxs, torch.tensor([max_workload_idx], device=self.device)])
        self.past_query_idxs = torch.cat([self.past_query_idxs, max_query_idxs])
        return max_query_idxs

    def _measure(self, answers):
        noisy_answers = gaussian_mech(answers, (1 - self.alpha) * self.eps0, 2 ** 0.5 * self.qm.sensitivity)
        noisy_answers = torch.clip(noisy_answers, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, noisy_answers])

class IterAlgoSingleRAP(IterAlgoRAPBase):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        max_query_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs = torch.cat([self.past_query_idxs, max_query_idx])
        return max_query_idx

    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = torch.clip(noisy_answer, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor([noisy_answer], device=self.device)])