import torch
import numpy as np

from tqdm import tqdm
from torch import optim

from src.qm import KWayMarginalQM
from src.algo.base import IterativeAlgorithmTorch
from src.utils.mechanisms import exponential_mech, gaussian_mech

class IterAlgoGEMBase(IterativeAlgorithmTorch):
    def __init__(self, G, T, eps0,
                 alpha=0.5, default_dir=None, verbose=False, seed=None,
                 loss_p=1, lr=1e-4, eta_min=1e-5, max_idxs=100, max_iters=100,
                 ema_beta=0.5, ema_error_factor=0.5,
                 ema_weights=True, ema_weights_beta=0.9,
                 save_interval=10, save_num=None,
                 ):
        super().__init__(G, T, eps0, alpha=alpha,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        # learning parameters
        self.loss_p = loss_p
        self.lr = lr
        self.eta_min = eta_min
        self.max_idxs = max_idxs
        self.max_iters = max_iters

        self.ema_beta = ema_beta
        self.ema_error_factor = ema_error_factor
        self.ema_weights = ema_weights
        self.ema_weights_beta = ema_weights_beta
        self.ema_error = None

        # saving
        self.save_interval = save_interval
        self.save_num = save_num
        if self.save_num is None:
            k_thresh = np.round(self.T * 0.5).astype(int)
            k_thresh = np.maximum(1, k_thresh)
            self.save_num = k_thresh

        self.optimizerG = optim.Adam(self.G.generator.parameters(), lr=self.lr)
        self.schedulerG = None
        if self.eta_min is not None:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.T, eta_min=self.eta_min)

    def _valid_qm(self):
        return (KWayMarginalQM)

    def _update_ema_error(self, error):
        if self.ema_error is None:
            self.ema_error = error
        self.ema_error = self.ema_beta * self.ema_error + (1 - self.ema_beta) * error

    def _get_loss(self, idxs):
        errors = self._get_sampled_query_errors(idxs=idxs)
        loss = torch.norm(errors, p=self.loss_p) / len(errors)
        return loss

    def _optimize_past_queries(self):
        threshold = self.ema_error_factor * self.ema_error

        lr = self.optimizerG.param_groups[0]['lr']
        optimizer = optim.Adam(self.G.generator.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_iters, eta_min=1e-8)

        loss = torch.tensor([0])
        step = 0
        for step in range(self.max_iters):
            optimizer.zero_grad()

            with torch.no_grad():
                errors = self._get_sampled_query_errors().abs()
            idxs = torch.arange(len(errors))

            # above THRESHOLD
            mask = errors >= threshold
            idxs, errors = idxs[mask], errors[mask]

            # get top MAX_IDXS
            max_errors_idxs = errors.argsort()[-self.max_idxs:]
            idxs, errors = idxs[max_errors_idxs], errors[max_errors_idxs]

            if len(idxs) == 0:  # no errors above threshold
                break

            loss = self._get_loss(idxs)
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss, step

    def fit(self, true_answers):
        if self.verbose:
            print("Fitting to query answers...")
        self.optimizerG.step() # just to avoid warning

        syn_answers = self.G.get_qm_answers()
        scores = (true_answers - syn_answers).abs()

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6}".format(scores.max().item()))

            # SAMPLE
            max_query_idx = self._sample(scores)

            # MEASURE
            self._measure(true_answers[max_query_idx])

            with torch.no_grad():
                errors = self._get_sampled_query_errors().abs()
            self.sampled_max_errors.append(errors.max().item())
            self._update_ema_error(self.sampled_max_errors[-1])

            loss, step = self._optimize_past_queries()

            syn_answers = self.G.get_qm_answers()
            scores = (true_answers - syn_answers).abs()
            self.record_errors(true_answers, syn_answers)

            if ((t + 1) % self.save_interval == 0) or (t + 1 > self.T - self.save_num):
                self.save('epoch_{}.pt'.format(t + 1))
            if np.min(self.true_max_errors) == self.true_max_errors[-1]:
                self.save('best.pt')
            self.save('last.pt')

            if self.schedulerG is not None:
                self.schedulerG.step()

        if self.ema_weights:
            self._ema_weights()

    # Inspired by https://arxiv.org/abs/1806.04498
    def _ema_weights(self):
        if self.verbose:
            print("Loading EMA of generator weights...")

        weights = {}
        k_array = np.arange(self.save_num)[::-1]
        for i in k_array:
            self.load('epoch_{}.pt'.format(self.T - i))
            w = self.G.generator.state_dict()
            for key in w.keys():
                if key not in weights.keys():
                    weights[key] = w[key]
                else:
                    weights[key] = self.ema_weights_beta * weights[key] + (1 - self.ema_weights_beta) * w[key]

        self.G.generator.load_state_dict(weights)

class IterAlgoGEM(IterAlgoGEMBase):
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

class IterAlgoSingleGEM(IterAlgoGEMBase):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        max_query_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs = torch.cat([self.past_query_idxs, max_query_idx])
        return max_query_idx

    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = torch.clip(noisy_answer, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor([noisy_answer], device=self.device)])