import argparse

from algorithms.mwem import MWEM
from qm import KWayMarginalSupportQMPublic

class PMWPub(MWEM):
    def _valid_qm(self):
        return (KWayMarginalSupportQMPublic)

def get_args():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, default='acs_PA')
    parser.add_argument('--dataset_pub', type=str, default='acs_OH')
    parser.add_argument('--marginal', type=int, default=3)
    parser.add_argument('--workload', type=int, default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    # privacy args
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=1.0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    # general algo args
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test_seed', type=int, default=None)

    # MWEM specific params
    parser.add_argument('--recycle', action='store_true', help='reuse past queries for MW')

    args = parser.parse_args()

    print(args)
    return args