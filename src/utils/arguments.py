import argparse

def get_args(base, iterative, public=False):
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--marginal', type=int, default=3)
    parser.add_argument('--workload', type=int, default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    # privacy args
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=1.0)
    parser.add_argument('--T', type=float, default=10.0)
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
    # misc
    parser.add_argument('--num_samples', type=int, default=100000)

    if base == 'nn':
        parser.add_argument('--K', type=int, default=1000)
        parser.add_argument('--dim', type=int, default=16)
        parser.add_argument('--gen_dim', type=int, default=256)
        parser.add_argument('--resample', action='store_true')
    elif base == 'fixed':
        parser.add_argument('--K', type=int, default=1000)
    elif base == 'nhist':
        pass
    else:
        assert False, 'invalid syndata generator selection'

    if iterative == 'iter':
        pass
    elif iterative == 'gem':
        parser.add_argument('--ema_weights', action='store_true')
        parser.add_argument('--ema_weights_beta', type=float, default=0.9)
    elif iterative == 'rap':
        parser.add_argument('--samples_per_round', type=int, default=1)
    elif iterative == 'non_dp':
        parser.add_argument('--sample_by_error', action='store_true')
        parser.add_argument('--log_freq', type=int, default=0)
        parser.add_argument('--save_all', action='store_true')
        parser.add_argument('--save_best', action='store_true')
    elif iterative == 'mwem':
        parser.add_argument('--recycle', action='store_true')
    elif iterative == 'pep':
        pass
    else:
        assert False, 'invalid iterative algorithm procedure'

    if public:
        parser.add_argument('--dataset_pub', type=str)

    args = parser.parse_args()

    if args.T.is_integer() and args.T != 1:
        args.T = int(args.T)

    print(args)
    return args

def get_T(T, workloads):
    if isinstance(T, float):
        T = int(len(workloads) * T)
    assert T <= len(workloads), "Number of iterations T must be <= total # workloads"
    assert T > 0, "T must be greater than 0"
    return T