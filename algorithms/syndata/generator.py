import torch
import argparse
from abc import ABC, abstractmethod
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, Embedding
from torch.nn import LeakyReLU, Sigmoid, LayerNorm

from utils.utils_data import Dataset
from utils.transformer import DataTransformer, get_domain_rows

import pdb

class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.out_dim = i + o
        self.fc = Linear(i, o)
        self.norm = BatchNorm1d(o)
        self.activation = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.norm(out)
        out = self.activation(out)
        return torch.cat([out, input], dim=1)

class GenerativeNetwork(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim, init_seed):
        super(GenerativeNetwork, self).__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)

        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim = seq[-1].out_dim
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

class Fixed(Module):
    def __init__(self, K, data_dim, init_seed):
        super(Fixed, self).__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)

        self.syndata = Embedding(K, data_dim)

    def forward(self, input):
        return self.syndata.weight

class Generator(ABC):
    def __init__(self, qm,
                 cont_columns=[], agg_mapping={},
                 K=1000, query_bs=10000, device=None, init_seed=None,
                 ):
        self.qm = qm
        self.agg_mapping = agg_mapping
        self.K = K
        self.query_bs = query_bs
        self.device = torch.device("cpu") if device is None else device
        self.init_seed = init_seed

        self.queries = self.qm.queries
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
        return answers

    def _get_onehot(self, data, how='sample'):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            if item[1] == 'softmax':
                probs = data[:, st:ed]
                out = torch.zeros_like(probs)
                if how == "sample":
                    idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
                elif how == "max":
                    idxs = probs.argmax(-1)
                else:
                    raise NotImplementedError
                out[torch.arange(out.shape[0]).to(self.device), idxs] = 1
                data_t.append(out)
            else:
                raise NotImplementedError
            st = ed
        return torch.cat(data_t, dim=1)

    def get_syndata(self, num_samples=100000, how='sample'): # TODO: any # samples
        samples = []

        syn = self.generate().detach()
        for i in range(num_samples // self.K):
            x = self._get_onehot(syn, how=how)
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
        self.generator = Fixed(self.K, self.data_dim, self.init_seed).to(self.device)

    def _generate(self):
        return self.generator(None)

class NeuralNetworkGenerator(Generator):
    def __init__(self, qm,
                 cont_columns=[], agg_mapping={},
                 K=1000, query_bs=10000, device=None, init_seed=None,
                 embedding_dim=128, gen_dims=None, resample=False,
                 ):
        self.embedding_dim = embedding_dim
        self.gen_dims = [2 * embedding_dim, 2 * embedding_dim] if gen_dims is None else gen_dims
        super().__init__(qm, cont_columns, agg_mapping, K, query_bs, device, init_seed=init_seed)

        self.resample = resample
        self.z_mean = torch.zeros(self.K, self.embedding_dim, device=self.device)
        self.z_std = torch.ones(self.K, self.embedding_dim, device=self.device)
        self.z = torch.normal(mean=self.z_mean, std=self.z_std)

    def _setup_generator(self):
        self.generator = GenerativeNetwork(self.embedding_dim, self.gen_dims, self.data_dim, self.init_seed).to(self.device)

    def _generate(self):
        if self.resample:
            self.z = torch.normal(mean=self.z_mean, std=self.z_std)
        return self.generator(self.z)

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