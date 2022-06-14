import argparse

def get_args(base, iterative):
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, default='adult')
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
    # iterative procedure args
    parser.add_argument('--loss_p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_idxs', type=int, default=100)

    if base == 'nn':
        parser.add_argument('--K', type=int, default=1000)
        parser.add_argument('--dim', type=int, default=512)
        parser.add_argument('--resample', action='store_true')
    elif base == 'fixed':
        parser.add_argument('--K', type=int, default=1000)

    if iterative == 'gem':
        parser.add_argument('--ema_weights', action='store_true')
        parser.add_argument('--ema_weights_beta', type=float, default=0.9)
    elif iterative == 'rap_softmax':
        parser.add_argument('--samples_per_round', type=int, default=1)

    args = parser.parse_args()

    print(args)
    return args