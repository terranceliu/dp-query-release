import numpy as np
from src.utils import Dataset

class NormalizedHistogram():
    def __init__(self, qm):
        self.qm = qm
        self.data_support = self.qm.data_support
        self._initialize_A()

    def _initialize_A(self):
        A_init = np.ones(len(self.data_support))
        A_init /= len(A_init)
        self.A = A_init
        self.A_avg = self.A.copy()

    def get_answers(self, idxs=None, use_avg=False):
        A = self.A_avg if use_avg else self.A
        answers = self.qm.get_answers(A)
        if idxs is not None:
            return answers[idxs]
        return answers

    def get_syndata(self, num_samples=100000, use_avg=False):
        A = self.A_avg if use_avg else self.A
        df_support = self.data_support.df
        idxs = np.random.choice(df_support.index.values, p=A, size=num_samples, replace=True)
        df_syn = df_support.loc[idxs].reset_index(drop=True)
        data_syn = Dataset(df_syn, self.data_support.domain)
        return data_syn