import torch
from src.qm import KWayMarginalQMTorch
from src.utils import get_args, get_data, get_rand_workloads, get_errors, save_results
from src.utils import get_per_round_budget_zCDP
from src.syndata.generator import NeuralNetworkGenerator
from src.algo.gem import IterAlgoGEM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args(base='nn', iterative='gem')

data = get_data(args.dataset)

workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQMTorch(data, workloads, device=device)

delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

model_save_dir = './save/GEM/{}/{}_{}_{}/{}_{}_{}_{}_{}/'.format(args.dataset,
                                                                 args.marginal, args.workload, args.workload_seed,
                                                                 args.epsilon, args.T, args.alpha,
                                                                 args.K, args.resample)

G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=None, resample=args.resample)
algo = IterAlgoGEM(G, args.T, eps0,
                   alpha=args.alpha, default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                   loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                   max_idxs=args.max_idxs, max_iters=args.max_iters,
                   ema_weights=args.ema_weights, ema_weights_beta=args.ema_weights_beta)
true_answers = query_manager.get_answers(data)
algo.fit(true_answers)

# get answers using sampled rows (alternatively you can get answers using G.get_qm_answers())
syndata = G.get_syndata(args.num_samples)
syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)
print(errors)

save_results("gem.csv", './results', args, errors)