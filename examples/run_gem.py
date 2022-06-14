import torch

from qm import KWayMarginalQM
from utils.arguments import get_args
from utils.utils_data import get_data, get_rand_workloads, get_default_cols
from utils.utils_general import get_errors, get_per_round_budget_zCDP

from algorithms.base.generator import NeuralNetworkGenerator
from algorithms.gem import IterativeAlgoGEM

args = get_args(base='nn', iterative='gem')

data = get_data(args.dataset)
data = data.project(get_default_cols(args.dataset))
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQM(data, workloads)

delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_dir = './save/GEM/{}/{}_{}_{}/{}_{}_{}_{}/'.format(args.dataset,
                                                              args.marginal, args.workload, args.workload_seed,
                                                              args.epsilon, args.T, args.alpha, args.K)

G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=None, resample=args.resample)
algo = IterativeAlgoGEM(G, query_manager, args.T, eps0, device,
                        alpha=args.alpha, default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                        loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                        max_idxs=args.max_idxs, max_iters=args.max_iters,
                        ema_error_factor=0.5, ema_weights=args.ema_weights, ema_weights_beta=args.ema_weights_beta)

true_answers = query_manager.get_answers(data)
algo.fit(true_answers)

syndata = algo.get_syndata()
syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)

print(errors)