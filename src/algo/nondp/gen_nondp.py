import torch
import numpy as np

from tqdm import tqdm
from torch import optim

from src.qm import KWayMarginalQMTorch, KWayMarginalSetQMTorch
from src.algo.base import IterativeAlgorithmTorch

class IterativeAlgoNonDP(IterativeAlgorithmTorch):
    def __init__(self, G, T,
                 loss_p=2, lr=1e-4, eta_min=1e-5, max_idxs=10000, max_iters=1,
                 sample_by_error=False, log_freq=0, save_all=False, save_best=False,
                 default_dir=None, verbose=False, seed=None,
                 ):
        super().__init__(G, T, eps0=0, alpha=0,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        self.loss_p = loss_p
        self.lr = lr
        self.eta_min = eta_min
        self.max_idxs = max_idxs
        self.max_iters = max_iters

        self.sample_by_error = sample_by_error
        self.log_freq = 1 if sample_by_error else log_freq
        if sample_by_error and log_freq != 1:
            print("sample_by_error=True -> defaulting log_freq to 1")
        self.save_all = save_all
        self.save_best = save_best
        assert log_freq >= 0, 'record_all_errors must be >= 0'
        if log_freq == 0:
            assert not save_best, 'save_best=True requires record_all_errors > 0'

        self.optimizerG = optim.Adam(self.G.generator.parameters(), lr=self.lr)
        self.schedulerG = None
        if self.eta_min is not None:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.T, eta_min=self.eta_min)

    def _valid_qm(self):
        return (KWayMarginalQMTorch, KWayMarginalSetQMTorch)

    def _sample(self, scores):
        pass

    def _measure(self, answers):
        pass

    def _get_loss(self, idxs):
        errors = self._get_sampled_query_errors(idxs=idxs)
        loss = torch.norm(errors, p=self.loss_p) / len(errors)
        return loss

    def fit(self, true_answers):
        if self.verbose:
            print('Fitting to query answers...')
        self.past_query_idxs = torch.arange(self.qm.num_queries)
        self.past_measurements = true_answers.clone()

        syn_answers = self.G.get_qm_answers()
        errors = (true_answers - syn_answers).abs()
        p = errors / errors.sum()

        pbar = tqdm(range(self.T)) if self.verbose else range(self.T)
        for t in pbar:
            if self.verbose and self.log_freq > 0:
                pbar.set_description('Max Error: {:.6}'.format(errors.max()))

            # if t > 200:
            #     errors = (true_answers - syn_answers).abs()
            #     print(errors.abs().sort()[0][:-10])
            #     print(errors.abs().max().item())
            #     print(errors.abs().mean().item())
            #     import pdb
            #     pdb.set_trace()

            if self.sample_by_error:
                for _ in range(self.max_iters):
                    self.optimizerG.zero_grad()
                    idxs = torch.multinomial(p, num_samples=self.max_idxs, replacement=True)
                    loss = self._get_loss(idxs)
                    loss.backward()
                    self.optimizerG.step()
            else:
                idxs_all = torch.randperm(len(errors))
                for idxs in torch.split(idxs_all, self.max_idxs):
                    self.optimizerG.zero_grad()
                    loss = self._get_loss(idxs)
                    loss.backward()
                    self.optimizerG.step()

            if self.schedulerG is not None:
                self.schedulerG.step()

            if self.log_freq > 0 and t % self.log_freq == 0:
                syn_answers = self.G.get_qm_answers()
                errors = (true_answers - syn_answers).abs()
                p = errors / errors.sum()
                self.record_errors(true_answers, syn_answers)
                if self.save_best and np.min(self.true_max_errors) == self.true_max_errors[-1]:
                    self.save('best.pt')

            if self.save_all:
                self.save('last.pt')

        self.save('last.pt')