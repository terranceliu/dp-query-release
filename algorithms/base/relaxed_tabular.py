import torch
import numpy as np
from torch.nn import Embedding

class UnifInitializer(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = (torch.rand(w.shape) - 0.5) * (1 / 0.5)

class WeightClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)
            module.weight.data = w

class RelaxedTabular():
    def __init__(self,
                 device, qm, data,
                 n, softmax
                 ):
        self.device = device
        self.queries = torch.tensor(qm.queries).to(self.device).long()

        self.syndata = Embedding(qm.dim, n)
        self.syndata.apply(UnifInitializer())
        self.syndata = self.syndata.to(self.device)
        self.softmax = softmax
        self.clipper = WeightClipper()

        self.domain = data.domain
        self._setup_domain()

    def clip_weights(self):
        if not self.softmax:
            self.syndata.apply(self.clipper)

    def _setup_domain(self):
        x = self.domain.shape
        x = np.cumsum(x)
        x = np.concatenate(([0], x))
        x = np.stack([x[:-1], x[1:]]).T
        self.query_attr_bin = x

        self.query_attr_dict = {}
        for i in np.arange(self.query_attr_bin.max()):
            self.query_attr_dict[i] = np.argmax([(i >= _x[0] and i < _x[1]) for _x in x])

    def _get_probs(self, x):
        data_t = []
        for i in range(self.query_attr_bin.shape[0]):
            bin = self.query_attr_bin[i]
            logits = x[:, bin[0]:bin[1]]
            probs = logits.softmax(-1)
            data_t.append(probs)
        return torch.cat(data_t, dim=1)

    def get_all_qm_answers(self):
        _x = self.syndata.weight.T.detach()
        if self.softmax:
            _x = self._get_probs(_x)

        out = []
        for queries in torch.split(self.queries, 10000):
            x = _x[:, queries]
            x = x.prod(dim=-1)
            x = x.mean(dim=0)
            out.append(x)
        out = torch.cat(out)
        return out

    def get_syndata(self):
        x = self.syndata.weight.T
        if self.softmax:
            x = self._get_probs(x)
        return x

    def get_distr_answers(self):
        answers = self.get_all_qm_answers()
        return answers.cpu().numpy()