import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch import optim

from qm import KWayMarginalQM
from algorithms.algo import IterativeAlgorithmTorch
from utils.mechanisms import exponential_mech, gaussian_mech
from algorithms.base.generative import GenerativeNetwork

class BaseGEM(IterativeAlgorithmTorch):
    def __init__(self, qm, T, eps0,
                 data, device,
                 alpha=0.5, default_dir=None,
                 cont_columns=[],
                 embedding_dim=128, gen_dim=(256, 256), batch_size=500, loss_p=1,
                 lr=1e-4, eta_min=1e-5, resample=False,
                 ema_beta=0.5, max_idxs=100, max_iters=100,
                 ema_error_factor=0, ema_weights=True, ema_weights_beta=0.9,
                 save_interval=10, save_num=None,
                 verbose=False, seed=None,
                 ):

        super().__init__(qm, T, eps0, alpha=alpha, default_dir=default_dir, verbose=verbose, seed=seed)
        self.G = GenerativeNetwork(device, qm, data, cont_columns=cont_columns,
                                   embedding_dim=embedding_dim, gen_dim=gen_dim,
                                   batch_size=batch_size, resample=resample)

        self.device = device
        self.queries = torch.tensor(self.qm.queries).to(self.device).long()
        self.loss_p = loss_p

        # learning parameters
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

    def _get_sampled_query_answers(self, fake_data):
        q_t_idxs = self.past_query_idxs.clone()
        fake_query_attr = fake_data[:, self.queries[q_t_idxs]]
        past_fake_answers = fake_query_attr.prod(-1).mean(axis=0)
        return past_fake_answers

    def _get_sampled_query_errors(self, fake_data):
        past_fake_answers = self._get_sampled_query_answers(fake_data)
        errors = past_fake_answers - self.past_measurements.to(self.device)
        errors = torch.clamp(errors.abs(), 0, np.infty)
        return errors

    def _update_ema_error(self, error):
        if self.ema_error is None:
            self.ema_error = error
        self.ema_error = self.ema_beta * self.ema_error + (1 - self.ema_beta) * error

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

    def _get_loss(self, fake_data, q_t_idxs, idxs):
        fake_query_attr = fake_data[:, self.queries[q_t_idxs]]
        fake_answer = fake_query_attr.prod(-1).mean(axis=0)
        real_answer = self.past_measurements[idxs].to(self.device)
        loss = real_answer - fake_answer
        loss = torch.norm(loss, p=self.loss_p) / len(loss)
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

            q_t_idxs = self.past_query_idxs.clone()
            idxs = torch.arange(q_t_idxs.shape[0])

            fake_data = self.G.generate_fake_data()
            errors = self._get_sampled_query_errors(fake_data.detach())

            # above THRESHOLD
            mask = errors >= threshold
            idxs, q_t_idxs, errors = idxs[mask], q_t_idxs[mask], errors[mask]

            # get top MAX_IDXS
            max_errors_idxs = errors.argsort()[-self.max_idxs:]
            idxs, q_t_idxs, errors = idxs[max_errors_idxs], q_t_idxs[max_errors_idxs], errors[max_errors_idxs]

            if len(q_t_idxs) == 0:  # no errors above threshold
                break

            loss = self._get_loss(fake_data, q_t_idxs, idxs)
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss, step

    def fit(self, true_answers):
        true_answers = torch.tensor(true_answers).to(self.device)
        fake_data = self.G.generate_fake_data()
        fake_answers = self.G.get_all_qm_answers(fake_data)
        scores = (true_answers - fake_answers).abs()

        for t in tqdm(range(self.T)):
            scores[self.past_query_idxs] = -np.infty # to ensure we don't resample past queries (though unlikely)

            # SAMPLE
            max_query_idx = self._sample(scores)

            # MEASURE
            self._measure(true_answers[max_query_idx])

            errors = self._get_sampled_query_errors(fake_data.detach())
            self.sampled_max_errors.append(errors.max().item())
            self._update_ema_error(self.sampled_max_errors[-1])

            loss, step = self._optimize_past_queries()

            fake_data = self.G.generate_fake_data()
            fake_answers = self.G.get_all_qm_answers(fake_data)
            scores = (true_answers - fake_answers).abs()

            self.record_errors(true_answers.cpu().numpy(), fake_answers.cpu().numpy())

            if ((t + 1) % self.save_interval == 0) or (t + 1 > self.T - self.save_num):
                self.save('epoch_{}.pkl'.format(t + 1))

            if self.verbose and step > 0:
                print("Epoch {}:\tTrue Error: {:.4f}\tEM Error: {:.4f}\n"
                      "Iters: {}\tLoss: {:.8f}".format(
                    t, self.true_max_errors[-1], self.sampled_max_errors[-1], step, loss.item()))

            if self.schedulerG is not None:
                self.schedulerG.step()

        if self.ema_weights:
            self._ema_weights()

    def get_syndata(self, num_samples=100000):
        return self.G.get_syndata(num_samples=num_samples)

    def get_answers(self):
        return self.G.get_distr_answers()

class GEM_Queries(BaseGEM):
    def _sample(self, scores):
        max_query_idx = exponential_mech(scores.detach().cpu().numpy(), self.alpha * self.eps0, self.qm.sensitivity)
        self.past_query_idxs = torch.cat([self.past_query_idxs, torch.tensor([max_query_idx])])
        return max_query_idx

    def _measure(self, answers):
        noisy_answer = gaussian_mech(answers, (1 - self.alpha) * self.eps0, self.qm.sensitivity)
        noisy_answer = torch.clamp(noisy_answer, 0, 1)
        self.past_measurements = torch.cat([self.past_measurements, torch.tensor([noisy_answer])])

class GEM_Workloads(BaseGEM):
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