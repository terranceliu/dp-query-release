import pdb

import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch import optim

from qm import KWayMarginalQM
from algorithms.base.algo import IterativeAlgorithmTorch
from utils.mechanisms import exponential_mech, gaussian_mech
from algorithms.base.generator import NeuralNetworkGenerator, FixedGenerator

class BaseGEM(IterativeAlgorithmTorch):
    def __init__(self, qm, T, eps0, device,
                 alpha=0.5, default_dir=None,
                 cont_columns=[], agg_mapping={},
                 embedding_dim=128, gen_dims=None, K=1000, loss_p=1,
                 lr=1e-4, eta_min=1e-5, resample=False,
                 ema_beta=0.5, max_idxs=100, max_iters=100,
                 ema_error_factor=0, ema_weights=True, ema_weights_beta=0.9,
                 save_interval=10, save_num=None,
                 verbose=False, seed=None,
                 ):

        super().__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.G = NeuralNetworkGenerator(qm, cont_columns=cont_columns, agg_mapping=agg_mapping, K=K, device=device,
                                        embedding_dim=embedding_dim, gen_dims=gen_dims, resample=resample,
                                        )

        self.device = device
        self.queries = torch.tensor(self.qm.queries).to(self.device).long()

        # learning parameters
        self.loss_p = loss_p
        self.lr = lr
        self.eta_min = eta_min
        self.ema_beta = ema_beta
        self.max_idxs = max_idxs
        self.max_iters = max_iters

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

    def _get_sampled_query_errors(self, idxs=None):
        q_t_idxs = self.past_query_idxs.clone()
        real_answers = self.past_measurements.to(self.device)
        if idxs is not None:
            q_t_idxs = q_t_idxs[idxs]
            real_answers = real_answers[idxs]

        syn = self.G.generate()
        syn_answers = self.G.get_answers(syn, idxs=q_t_idxs)
        errors = real_answers - syn_answers
        return errors

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
        self.optimizerG.step() # just to avoid warning

        syn_answers = self.G.get_qm_answers()
        scores = np.abs(true_answers - syn_answers)

        pbar = tqdm(range(self.T))
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6}".format(scores.max().item()))

            # SAMPLE
            max_query_idx = self._sample(scores)

            # MEASURE
            self._measure(true_answers[max_query_idx])

            errors = self._get_sampled_query_errors().abs()
            self.sampled_max_errors.append(errors.max().item())
            self._update_ema_error(self.sampled_max_errors[-1])

            loss, step = self._optimize_past_queries()

            syn_answers = self.G.get_qm_answers()
            scores = np.abs(true_answers - syn_answers)
            self.record_errors(true_answers, syn_answers)

            if ((t + 1) % self.save_interval == 0) or (t + 1 > self.T - self.save_num):
                self.save('epoch_{}.pkl'.format(t + 1))
            if np.min(self.true_max_errors) == self.true_max_errors[-1]:
                self.save('best.pkl')
            self.save('last.pkl')

            if self.schedulerG is not None:
                self.schedulerG.step()

        if self.ema_weights:
            self._ema_weights()

    def get_syndata(self, num_samples=100000):
        return self.G.get_syndata(num_samples=num_samples)

    def get_answers(self):
        return self.G.get_qm_answers()

    # Inspired by https://arxiv.org/abs/1806.04498
    def _ema_weights(self):
        if self.verbose:
            print("Loading EMA of generator weights...")

        weights = {}
        k_array = np.arange(self.save_num)[::-1]
        for i in k_array:
            self.load('epoch_{}.pkl'.format(self.T - i))
            w = self.G.generator.state_dict()
            for key in w.keys():
                if key not in weights.keys():
                    weights[key] = w[key]
                else:
                    weights[key] = self.ema_weights_beta * weights[key] + (1 - self.ema_weights_beta) * w[key]

        self.G.generator.load_state_dict(weights)

class GEM(BaseGEM):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        max_query_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor([max_query_idx])])
        return max_query_idx

    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = np.clip(noisy_answer, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor([noisy_answer])])

class GEM_Marginal(BaseGEM):
    def _sample(self, scores):
        scores[self.past_query_idxs] = -np.infty
        scores = list(scores[x[0]:x[1]] for x in list(self.qm.workload_idxs))
        scores = np.array([x.max() for x in scores]) # get max workload scores
        max_workload_idx = exponential_mech(scores, self.alpha * self.eps0, self.qm.sensitivity)
        max_query_idxs = np.arange(*self.qm.workload_idxs[max_workload_idx])

        self.past_workload_idxs = torch.cat([self.past_workload_idxs, torch.tensor([max_workload_idx])])
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor(max_query_idxs)])
        return max_query_idxs

    def _measure(self, answers):
        noisy_answers = gaussian_mech(answers, (1 - self.alpha) * self.eps0, 2 ** 0.5 * self.qm.sensitivity)
        noisy_answers = np.clip(noisy_answers, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor(noisy_answers)])

class GEM_Nondp(BaseGEM):
    def __init__(self, qm, T, device, default_dir=None,
                 cont_columns=[], agg_mapping={},
                 embedding_dim=128, gen_dims=None, K=500, loss_p=1,
                 lr=1e-4, eta_min=1e-5, resample=False, ema_error_factor=0.5,
                 max_idxs=10000, max_iters=1,
                 verbose=False, seed=None,
                 ):
        super().__init__(qm, T, 0, device, default_dir=default_dir,
                         cont_columns=cont_columns, agg_mapping=agg_mapping,
                         embedding_dim=embedding_dim, gen_dims=gen_dims, K=K, loss_p=loss_p,
                         lr=lr, eta_min=eta_min, resample=resample, max_idxs=max_idxs, max_iters=max_iters,
                         ema_error_factor=ema_error_factor, verbose=verbose, seed=seed)

    def _sample(self, scores):
        pass

    def _measure(self, answers):
        pass

    def fit(self, true_answers):
        self.past_query_idxs = torch.arange(self.qm.num_queries)
        self.past_measurements = torch.tensor(true_answers)

        syn_answers = self.G.get_qm_answers()
        errors = np.abs(true_answers - syn_answers)

        pbar = tqdm(range(self.T))
        for t in pbar:
            if self.verbose:
                pbar.set_description("Max Error: {:.6}".format(errors.max()))

            p = errors / errors.sum()
            # print(self.qm.workloads[self.qm.query_workload_map[errors.argmax()]])

            for _ in range(self.max_iters):
                self.optimizerG.zero_grad()
                idxs = np.random.choice(len(true_answers), size=self.max_idxs, p=p, replace=True)
                loss = self._get_loss(idxs)

                loss.backward()
                self.optimizerG.step()

            if self.schedulerG is not None:
                self.schedulerG.step()

            syn_answers = self.G.get_qm_answers()
            errors = np.abs(true_answers - syn_answers)
            self.record_errors(true_answers, syn_answers)

            if np.min(self.true_max_errors) == self.true_max_errors[-1]:
                self.save('best.pkl')
            self.save('last.pkl')

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

    # GEM specific args
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--syndata_size', type=int, default=1000)
    parser.add_argument('--loss_p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_idxs', type=int, default=100)
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--ema_weights', action='store_true')
    parser.add_argument('--ema_weights_beta', type=float, default=0.9)

    args = parser.parse_args()

    print(args)
    return args