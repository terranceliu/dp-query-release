import sys
import torch
from src.qm import KWayMarginalQMTorch
from src.utils import get_args, get_data, get_rand_workloads, get_cached_true_answers, get_T, \
    get_errors, save_results, check_existing_results
from src.utils import get_per_round_budget_zCDP
from src.syndata import NeuralNetworkGenerator
from src.algo import IterAlgo

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = get_args(base='nn', iterative='iter')
results_fn = 'gem.csv'

data = get_data(args.dataset)

workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)
args.workload = len(workloads)

if check_existing_results(results_fn, args):
    sys.exit("Exiting... run previously completed")

query_manager = KWayMarginalQMTorch(data, workloads, verbose=args.verbose, device=device)
true_answers = get_cached_true_answers(args, data, query_manager)

T = get_T(args.T, workloads)
delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, T, alpha=args.alpha)

model_save_dir = './save/GEM/{}/{}_{}_{}/{}_{}_{}_{}_{}/'.format(args.dataset,
                                                                 args.marginal, args.workload, args.workload_seed,
                                                                 args.epsilon, args.T, args.alpha,
                                                                 args.K, args.resample)

G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=[args.gen_dim] * 2, resample=args.resample)
algo = IterAlgo(G, T, eps0,
                alpha=args.alpha, default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                loss_p=args.loss_p, lr=args.lr, max_iters=args.max_iters, max_idxs=args.max_idxs,
                )

algo.fit(true_answers)

if args.num_samples == 0:
    syndata_answers = G.get_qm_answers()
else:
    syndata = G.get_syndata(args.num_samples)
    syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)
print(errors)

save_results(results_fn, './results', args, errors)