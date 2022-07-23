from src.algo.pep import PEP
from src.qm import KWayMarginalSupportQMPublic

class PEPPub(PEP):
    def __init__(self, G, T, eps0,
                 alpha=0.5, max_iters=100,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(G, T, eps0, alpha=alpha, max_iters=max_iters,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        self.G.A = self.qm.histogram_public.copy()
        self.G.A_avg = self.qm.histogram_public.copy()

    def _valid_qm(self):
        return (KWayMarginalSupportQMPublic)