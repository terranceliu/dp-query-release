from src.algo.mwem import MWEMSingle
from src.qm import KWayMarginalSupportQMPublic

class PMWPubSingle(MWEMSingle):
    def __init__(self, G, T, eps0,
                 alpha=0.5, recycle_queries=False,
                 default_dir=None, verbose=False, seed=None):
        super().__init__(G, T, eps0, alpha=alpha, recycle_queries=recycle_queries,
                         default_dir=default_dir, verbose=verbose, seed=seed)
        self.G.A = self.qm.histogram_public.copy()
        self.G.A_avg = self.qm.histogram_public.copy()

    def _valid_qm(self):
        return (KWayMarginalSupportQMPublic)