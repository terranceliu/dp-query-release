import torch
import numpy as np

from tqdm import tqdm
from torch import optim

from qm import KWayMarginalQM
from algorithms.algo import IterativeAlgorithmTorch

class IterativeAlgoNonDP(IterativeAlgorithmTorch):
    def __init__(self, G, qm, T,
                 default_dir=None, verbose=False, seed=None,
                 loss_p=2, lr=1e-4, eta_min=1e-5, max_idxs=10000, max_iters=1,
                 ):
        super().__init__(G, qm, T, eps0=0, alpha=0,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        self.loss_p = loss_p
        self.lr = lr
        self.eta_min = eta_min
        self.max_idxs = max_idxs
        self.max_iters = max_iters

        self.optimizerG = optim.Adam(self.G.generator.parameters(), lr=self.lr)
        self.schedulerG = None
        if self.eta_min is not None:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.T, eta_min=self.eta_min)

    def _valid_qm(self):
        return (KWayMarginalQM)

    def _sample(self, scores):
        pass

    def _measure(self, answers):
        pass

    def _get_loss(self, idxs):
        errors = self._get_sampled_query_errors(idxs=idxs)
        loss = torch.norm(errors, p=self.loss_p) / len(errors)
        return loss

    def fit(self, true_answers):
        print("Fitting to query answers...")
        self.past_query_idxs = torch.arange(self.qm.num_queries)
        self.past_measurements = true_answers.clone()

        syn_answers = self.G.get_qm_answers()
        errors = (true_answers - syn_answers).abs()

        pbar = tqdm(range(self.T))
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6}".format(errors.max()))

            p = errors / errors.sum()
            for _ in range(self.max_iters):
                self.optimizerG.zero_grad()
                idxs = torch.multinomial(p, num_samples=self.max_idxs, replacement=True)
                loss = self._get_loss(idxs)

                loss.backward()
                self.optimizerG.step()

            if self.schedulerG is not None:
                self.schedulerG.step()

            syn_answers = self.G.get_qm_answers()
            errors = (true_answers - syn_answers).abs()
            self.record_errors(true_answers, syn_answers)

            if np.min(self.true_max_errors) == self.true_max_errors[-1]:
                self.save('best.pkl')
            self.save('last.pkl')