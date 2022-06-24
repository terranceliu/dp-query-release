from algorithms.mwem import MWEM
from qm import KWayMarginalSupportQMPublic

class PMWPub(MWEM):
    def _valid_qm(self):
        return (KWayMarginalSupportQMPublic)