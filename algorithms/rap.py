import torch
import argparse

import numpy as np

from tqdm import tqdm
from torch import optim

from qm import KWayMarginalQM
from algorithms.algo import IterativeAlgorithmTorch
from utils.mechanisms import exponential_mech, gaussian_mech

from algorithms.base.relaxed_tabular import RelaxedTabular

class RAPBase(IterativeAlgorithmTorch):
    def __init__(self, qm, T, eps0,
                 data, device,
                 alpha=0.5, default_dir=None,
                 n=1000, K=1,
                 softmax=False,
                 lr=1e-4, max_iters=5000, max_idxs=None,
                 verbose=False, seed=None):
        super(RAPBase, self).__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.rt = RelaxedTabular(device, qm, data, n, softmax)

        self.device = device
        self.queries = torch.tensor(self.qm.queries).to(self.device).long()
        self.K = K

        # learning parameters
        self.lr = lr
        self.max_iters = max_iters
        self.max_idxs = max_idxs

        self.optimizer = optim.Adam(self.rt.syndata.parameters(), lr=self.lr)

    def _valid_qm(self):
        return (KWayMarginalQM)

    def _get_sampled_query_answers(self, idxs=None):
        q_t_idxs = self.past_query_idxs.clone()
        if idxs is not None:
            q_t_idxs = q_t_idxs[idxs]
        x = self.rt.get_syndata()
        x = x[:, self.queries[q_t_idxs]]
        return x.prod(dim=-1).mean(dim=0)

    def _get_sampled_noisy_meausrements(self, idxs=None):
        x = self.past_measurements.to(self.device)
        if idxs is not None:
            x = x[idxs]
        return x

    def _get_sampled_query_errors(self):
        past_fake_answers = self._get_sampled_query_answers()
        errors = past_fake_answers - self.past_measurements.to(self.device)
        errors = torch.clamp(errors.abs(), 0, np.infty)
        return errors

    def _get_loss(self):
        idxs = None if self.max_idxs is None else torch.randperm(len(self.past_query_idxs))[:self.max_idxs]
        errors = self._get_sampled_query_answers(idxs) - self._get_sampled_noisy_meausrements(idxs)
        loss = torch.norm(errors, p=2) ** 2
        return loss

    def fit(self, true_answers):
        true_answers = torch.tensor(true_answers).to(self.device)
        fake_answers = self.rt.get_all_qm_answers()
        scores = (true_answers - fake_answers).abs()

        pbar = tqdm(range(self.T))
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6}".format(scores.max().item()))

            for _ in range(self.K):
                scores[self.past_query_idxs] = -np.infty
                max_query_idx = self._sample(scores)
                self._measure(true_answers[max_query_idx])

            step = 0
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.max_iters,
                                                             threshold=1e-7, threshold_mode='rel')  # scheduler is just used for early stopping

            while step < self.max_iters:
                self.optimizer.zero_grad()

                loss = self._get_loss()
                if loss < 1e-8:
                    break

                loss.backward()
                scheduler.step(loss)
                if scheduler.num_bad_epochs > 10:
                    break

                self.optimizer.step()
                self.rt.clip_weights()
                step += 1

            fake_answers = self.rt.get_all_qm_answers()
            scores = (true_answers - fake_answers).abs()

            self.record_errors(true_answers.cpu().numpy(), fake_answers.cpu().numpy())

    def get_syndata(self):
        pass

    def get_answers(self):
        return self.rt.get_distr_answers()

class RAP(RAPBase):
    def _sample(self, scores):
        max_query_idx = exponential_mech(scores.detach().cpu().numpy(), self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor([max_query_idx])])
        return max_query_idx

    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = torch.clamp(noisy_answer, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor([noisy_answer])])

class RAP_Marginal(RAPBase): # sensitivity trick for marginal queries
    def _sample(self, scores):
        scores = list(scores[x[0]:x[1]] for x in list(self.qm.workload_idxs))
        scores = np.array([x.max().item() for x in scores]) # get max workload scores
        max_workload_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        max_query_idxs = np.arange(*self.qm.workload_idxs[max_workload_idx])

        self.past_workload_idxs = torch.cat([self.past_workload_idxs, torch.tensor([max_workload_idx])])
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor(max_query_idxs)])
        return max_query_idxs

    def _measure(self, answers):
        noisy_answers = gaussian_mech(answers.cpu(), (1 - self.alpha) * self.eps0, 2 ** 0.5 * self.qm.sensitivity)
        noisy_answers = torch.clamp(noisy_answers, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, noisy_answers])

def get_args():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, default='adult')
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

    # RAP specific args
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--softmax', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--max_idxs', type=int, default=10000)

    args = parser.parse_args()

    print(args)
    return args
