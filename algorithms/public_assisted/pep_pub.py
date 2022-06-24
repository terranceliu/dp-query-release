from algorithms.pep import PEP
from qm import KWayMarginalSupportQMPublic

class PEPPub(PEP):
    def _valid_qm(self):
        return (KWayMarginalSupportQMPublic)