import torch
import argparse
import numpy as np
import pandas as pd

from torch import optim
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential

from utils.utils_data import Dataset
from utils.transformer import DataTransformer, get_missing_rows

import pdb

class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
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

class GenerativeNetwork():
    def __init__(self,
                 device, qm, data,
                 cont_columns=[],
                 embedding_dim=128, gen_dim=(256, 256),
                 batch_size=500, resample=False,
                 ):
        self.device = device
        self.qm = qm
        self.queries = torch.tensor(self.qm.queries).to(self.device).long()

        # network architecture
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.batch_size = batch_size
        self.resample = resample

        self.mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        self.std = self.mean + 1

        self.domain = data.domain
        discrete_columns = [col for col in data.df.columns.values if col not in cont_columns]
        self._setup_data(data.df, self.domain, discrete_columns=discrete_columns)

    def _setup_data(self, train_data, domain, discrete_columns=[], overrides=[]):
        extra_rows = get_missing_rows(train_data, discrete_columns, domain)
        if len(extra_rows) > 0:
            train_data = pd.concat([extra_rows, train_data]).reset_index(drop=True)

        if not hasattr(self, "transformer") or 'transformer' in overrides:
            self.transformer = DataTransformer()
            self.transformer.fit(train_data, discrete_columns)

        if not hasattr(self, "generator") or 'generator' in overrides:
            data_dim = self.transformer.output_dimensions
            self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
            if self.batch_size == 1: # can't apply batch norm if batch_size = 1
                self.generator.eval()

    def _apply_activate(self, data, tau=0.2):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            out = data[:, st:ed]
            if item[1] is None:
                pass
            elif item[1] == 'softmax':
                out = out.softmax(-1)
            elif item[1] == 'tanh':
                out = out.tanh()
            elif item[1] == 'sigmoid':
                out = 1 / (1 + torch.exp(-out / 5))
            else:
                raise NotImplementedError
            data_t.append(out)
            st = ed
        return torch.cat(data_t, dim=1)

    def generate_fake_data(self):
        if not hasattr(self, "fakez") or self.resample:
            self.fakez = torch.normal(mean=self.mean, std=self.std)
        fake = self.generator(self.fakez)
        fake_data = self._apply_activate(fake)
        return fake_data

    def get_all_qm_answers(self, fake_data):
        queries = self.qm.queries
        fake_answers = torch.zeros(queries.shape[0]).to(self.device)
        for fake_data_chunk in torch.split(fake_data.detach(), 25):# 100  #TODO: make adaptive to fit GPU memory
            x = fake_data_chunk[:, queries]
            # mask = qm.queries < 0 # TODO: mask out -1 queries for different k-ways
            x = x.prod(-1)
            x = x.sum(axis=0)
            fake_answers += x
        fake_answers /= fake_data.shape[0]
        return fake_answers

    def get_onehot(self, data, how='sample'):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            if item[1] == 'softmax':
                probs = data[:, st:ed]
                out = torch.zeros_like(probs)
                if how == 'sample':
                    idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
                elif how == 'argmax':
                    idxs = probs.argmax(-1)
                else:
                    assert 0
                out[torch.arange(out.shape[0]).to(self.device), idxs] = 1
                data_t.append(out)
            else:
                assert 0
            st = ed
        return torch.cat(data_t, dim=1)

    def get_distr_answers(self):
        syn_distribution = self.generate_fake_data()
        answers = self.get_all_qm_answers(syn_distribution)
        return answers.cpu().numpy()

    def get_syndata(self, num_samples=100000):
        samples = []

        syn_distribution = self.generate_fake_data()
        for i in range(num_samples // self.batch_size):
            x = self.get_onehot(syn_distribution)
            samples.append(x)
        x = torch.cat(samples, dim=0).cpu()
        df = self.transformer.inverse_transform(x)
        data_synth = Dataset(df, self.domain)

        return data_synth

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