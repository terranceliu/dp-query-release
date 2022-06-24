import numpy as np

class ApproxDistr():
    def __init__(self, qm):
        self.qm = qm
        self.Gata_support = self.qm.data_support
        self._initialize_A()

    def _initialize_A(self):
        A_init = np.ones(len(self.Gata_support))
        A_init /= len(A_init)
        self.A = A_init
        self.A_avg = self.A.copy()

    def get_answers(self, idxs=None, use_avg=False):
        A = self.A_avg if use_avg else self.A
        answers = self.qm.get_answers(A)
        if idxs is not None:
            return answers[idxs]
        return answers

    # TODO
    def get_syndata(self, num_samples=100000, use_avg=False):
        A = self.A_avg if use_avg else self.A
        return A