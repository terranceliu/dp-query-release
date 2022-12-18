import torch
import numpy as np

from tqdm import tqdm
from torch import optim

from src.qm import KWayMarginalQM
from src.algo.base import IterativeAlgorithmTorch
from src.utils.mechanisms import exponential_mech, gaussian_mech

class IterAlgoBase(IterativeAlgorithmTorch):
    def __init__(self, G, T, eps0,
                 alpha=0.5, default_dir=None, verbose=False, seed=None,
                 loss_p=2, lr=1e-4, max_iters=1000, max_idxs=10000):
        super().__init__(G, T, eps0, alpha=alpha,
                         default_dir=default_dir, verbose=verbose, seed=seed)

        # learning parameters
        self.loss_p = loss_p
        self.lr = lr
        self.max_iters = max_iters
        self.max_idxs = max_idxs

        self.optimizer = optim.Adam(self.G.generator.parameters(), lr=self.lr)

    def _valid_qm(self):
        return (KWayMarginalQM)

    def _get_loss(self, idxs):
        errors = self._get_sampled_query_errors(idxs=idxs)
        loss = torch.norm(errors, p=self.loss_p) / len(errors)
        return loss

    def fit(self, true_answers):
        if self.verbose:
            print("Fitting to query answers...")
        syn_answers = self.G.get_qm_answers()
        scores = (true_answers - syn_answers).abs()

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.4} | Mean Error: {:.4}".format(scores.max().item(), scores.mean().item()))

            max_query_idx = self._sample(scores)
            self._measure(true_answers[max_query_idx])

            # pbar1 = tqdm(range(self.max_iters))
            # for step in pbar1:
            for step in range(self.max_iters):
                with torch.no_grad():
                    errors = self._get_sampled_query_errors().abs()

                # errors = scores[self.past_query_idxs]
                p = errors / errors.sum()
                # if self.verbose:
                #     pbar1.set_description('{} - Max Error: {:.6}'.format(len(errors), errors.max()))

                self.optimizer.zero_grad()
                idxs = torch.multinomial(p, num_samples=self.max_idxs, replacement=True)
                loss = self._get_loss(idxs)
                loss.backward()
                self.optimizer.step()

                # if (step + 1) % (self.max_iters / 10) == 0:
                #     syn_answers = self.G.get_qm_answers()

            syn_answers = self.G.get_qm_answers()
            scores = (true_answers - syn_answers).abs()
            self.record_errors(true_answers, syn_answers)

class IterAlgo(IterAlgoBase):
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