import torch
import argparse
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from torch import optim
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, Embedding

from utils.utils_data import Dataset
from utils.transformer import DataTransformer, get_domain_rows

import pdb

class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.activation = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.activation(out)
        return torch.cat([out, input], dim=1)

class GenerativeNetwork(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(GenerativeNetwork, self).__init__()

        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

class Fixed(Module):
    def __init__(self, K, data_dim):
        super(Fixed, self).__init__()
        self.syndata = Embedding(K, data_dim)

    def forward(self, input):
        return self.syndata.weight

class Generator(ABC):
    def __init__(self, qm,
                 cont_columns=[], agg_mapping={},
                 K=1000, query_bs=10000, device='cpu',
                 ):
        self.qm = qm
        self.agg_mapping = agg_mapping
        self.K = K
        self.query_bs = query_bs
        self.device = device

        self.queries = torch.tensor(self.qm.queries).to(self.device).long()
        self.domain = self.qm.domain
        self.cont_columns = cont_columns
        self.discrete_columns = [col for col in self.domain.attrs if col not in self.cont_columns]
        self._setup()

    @abstractmethod
    def _setup_generator(self):
        pass

    @abstractmethod
    def _generate(self):
        pass

    def _setup(self):
        df_domain = get_domain_rows(self.domain, self.discrete_columns)
        self.transformer = DataTransformer()
        self.transformer.fit(df_domain, self.discrete_columns)
        self.data_dim = self.transformer.output_dimensions
        self._setup_generator()

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            out = data[:, st:ed]
            if item[1] is None:
                pass
            elif item[1] == 'softmax':
                out = out.softmax(-1)
            else:
                raise NotImplementedError
            data_t.append(out)
            st = ed
        # data_t.append(data[:, [-1]].abs())
        return torch.cat(data_t, dim=1)

    def generate(self):
        x = self._generate()
        x = self._apply_activate(x)
        for pos, pos_agg_list in self.agg_mapping.items():
            x[:, pos] = x[:, pos_agg_list].sum(axis=-1)
        return x

    def get_answers(self, x, idxs=None):
        # fake_data, sampling_weights = fake_data[:, :-1], fake_data[:, -1:]
        queries = self.queries
        if idxs is not None:
            queries = queries[idxs]

        answers = []
        for queries_batch in torch.split(queries, self.query_bs, dim=0):
            answers_batch = x[:, queries_batch.T]
            answers_batch = answers_batch.prod(1)
            # answers_batch = answers_batch * sampling_weights
            # answers_batch = answers_batch / sampling_weights.sum()
            answers_batch = answers_batch / self.K
            answers_batch = answers_batch.sum(axis=0)
            answers.append(answers_batch)
        answers = torch.cat(answers)

        return answers

    def get_qm_answers(self):
        syn = self.generate().detach()
        answers = self.get_answers(syn)
        return answers.cpu().numpy()

    def _get_onehot(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            if item[1] == 'softmax':
                probs = data[:, st:ed]
                out = torch.zeros_like(probs)
                idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
                out[torch.arange(out.shape[0]).to(self.device), idxs] = 1
                data_t.append(out)
            else:
                raise NotImplementedError
            st = ed
        return torch.cat(data_t, dim=1)

    def get_syndata(self, num_samples=100000): # TODO: any # samples
        samples = []

        syn = self.generate().detach()
        for i in range(num_samples // self.K):
            x = self._get_onehot(syn)
            samples.append(x)
        samples = torch.cat(samples, dim=0).cpu()

        samples[:, list(self.agg_mapping.keys())] = 0
        for key, val in self.agg_mapping.items():
            mask = samples[:, val].max(-1)[0] == 1
            samples[mask, key] = 1

        df = self.transformer.inverse_transform(samples)
        data_synth = Dataset(df, self.domain)
        return data_synth

class FixedGenerator(Generator):
    def _setup_generator(self):
        self.generator = Fixed(self.K, self.data_dim).to(self.device)

    def _generate(self):
        return self.generator(None)

class NeuralNetworkGenerator(Generator):
    def __init__(self, qm,
                 cont_columns=[], agg_mapping={},
                 K=1000, query_bs=10000, device='cpu',
                 embedding_dim=128, gen_dims=None, resample=False,
                 ):
        self.embedding_dim = embedding_dim
        self.gen_dims = [2 * embedding_dim, 2 * embedding_dim] if gen_dims is None else gen_dims
        super().__init__(qm, cont_columns, agg_mapping, K, query_bs, device)

        self.resample = resample
        self.z_mean = torch.zeros(self.K, self.embedding_dim, device=self.device)
        self.z_std = torch.ones(self.K, self.embedding_dim, device=self.device)
        self.z = torch.normal(mean=self.z_mean, std=self.z_std)

    def _setup_generator(self):
        self.generator = GenerativeNetwork(self.embedding_dim, self.gen_dims, self.data_dim).to(self.device)

    def _generate(self):
        if self.resample:
            self.z = torch.normal(mean=self.z_mean, std=self.z_std)
        return self.generator(self.z)

# class Generator():
#     def __init__(self,
#                  device, qm,
#                  cont_columns=[], agg_mapping={},
#                  embedding_dim=128, gen_dim=None,
#                  batch_size=500, resample=False,
#                  qm_query_bs=10000
#                  ):
#         self.qm_query_bs = qm_query_bs
#
#         self.device = device
#         self.qm = qm
#         self.queries = torch.tensor(self.qm.queries).to(self.device).long()
#
#         # network architecture
#         self.embedding_dim = embedding_dim
#         self.gen_dim = [2 * embedding_dim, 2 * embedding_dim] if gen_dim is None else gen_dim
#         self.batch_size = batch_size
#         self.resample = resample
#
#         self.mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
#         self.std = self.mean + 1
#
#         self.domain = self.qm.domain
#         self.cont_columns = cont_columns
#         self.discrete_columns = [col for col in self.domain.attrs if col not in self.cont_columns]
#         self._setup_data()
#
#         self.agg_mapping = agg_mapping
#
#     def _setup_data(self, overrides=[]):
#         df_domain = get_domain_rows(self.domain, self.discrete_columns)
#
#         if not hasattr(self, "transformer") or 'transformer' in overrides:
#             self.transformer = DataTransformer()
#             self.transformer.fit(df_domain, self.discrete_columns)
#
#         if not hasattr(self, "generator") or 'generator' in overrides:
#             data_dim = self.transformer.output_dimensions
#             # self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
#             self.generator = Fixed(self.batch_size, data_dim).to(self.device)
#             if self.batch_size == 1: # can't apply batch norm if batch_size = 1
#                 self.generator.eval()
#
#     def _apply_activate(self, data, tau=0.2):
#         data_t = []
#         st = 0
#         for item in self.transformer.output_info:
#             ed = st + item[0]
#             out = data[:, st:ed]
#             if item[1] is None:
#                 pass
#             elif item[1] == 'softmax':
#                 out = out.softmax(-1)
#             elif item[1] == 'tanh':
#                 out = out.tanh()
#             elif item[1] == 'sigmoid':
#                 out = 1 / (1 + torch.exp(-out / 5))
#             else:
#                 raise NotImplementedError
#             data_t.append(out)
#             st = ed
#         # data_t.append(data[:, [-1]].abs())
#         return torch.cat(data_t, dim=1)
#
#     def generate_fake_data(self):
#         if not hasattr(self, "fakez") or self.resample:
#             self.fakez = torch.normal(mean=self.mean, std=self.std)
#         fake = self.generator(self.fakez)
#         fake_data = self._apply_activate(fake)
#
#         for pos, pos_agg_list in self.agg_mapping.items():
#             fake_data[:, pos] = fake_data[:, pos_agg_list].sum(axis=-1)
#         # assert torch.isclose(fake_data[:, self.qm.col_pos_map['GID_0']].sum(axis=-1),
#         #                      torch.tensor(1., device=self.device)).all()
#
#         return fake_data
#
#     def get_query_answers(self, fake_data, idxs=None):
#         # fake_data, sampling_weights = fake_data[:, :-1], fake_data[:, -1:]
#
#         queries = self.queries
#         if idxs is not None:
#             queries = queries[idxs]
#
#         answers = []
#         for queries_batch in torch.split(queries, self.qm_query_bs, dim=0):
#             x = fake_data[:, queries_batch.T]
#             x = x.prod(1)
#             # x = x * sampling_weights
#             # x = x / sampling_weights.sum()
#             x = x / len(fake_data)
#             x = x.sum(axis=0)
#             # x = x.mean(axis=0)
#             answers.append(x)
#         answers = torch.cat(answers)
#
#         return answers
#
#     def get_onehot(self, data, how='sample'):
#         data_t = []
#         st = 0
#         for item in self.transformer.output_info:
#             ed = st + item[0]
#             if item[1] == 'softmax':
#                 probs = data[:, st:ed]
#                 out = torch.zeros_like(probs)
#                 if how == 'sample':
#                     idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
#                 elif how == 'argmax':
#                     idxs = probs.argmax(-1)
#                 else:
#                     assert 0
#                 out[torch.arange(out.shape[0]).to(self.device), idxs] = 1
#                 data_t.append(out)
#             else:
#                 assert 0
#             st = ed
#         return torch.cat(data_t, dim=1)
#
#     def get_distr_answers(self):
#         syn = self.generate_fake_data().detach()
#         answers = self.get_query_answers(syn)
#         return answers.cpu().numpy()
#
#     def get_syndata(self, num_samples=100000):
#         samples = []
#
#         syn = self.generate_fake_data()
#         for i in range(num_samples // self.batch_size):
#             x = self.get_onehot(syn)
#             samples.append(x)
#         x = torch.cat(samples, dim=0).cpu()
#         x[:, list(self.agg_mapping.keys())] = 0
#         for key, val in self.agg_mapping.items():
#             mask = x[:, val].max(-1)[0] == 1
#             x[mask, key] = 1
#         df = self.transformer.inverse_transform(x)
#         data_synth = Dataset(df, self.domain)
#
#         return data_synth

def get_args():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    # privacy args
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=1.0)
    # general algo args
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test_seed', type=int, default=None)

    # GEM specific args
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--syndata_size', type=int, default=1000)
    parser.add_argument('--loss_p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_iters', type=int, default=1000000)
    parser.add_argument('--max_idxs', type=int, default=10000)
    parser.add_argument('--resample', action='store_true')

    args = parser.parse_args()

    print(args)
    return args