import torch
from src.qm import KWayMarginalQMTorch
from src.utils import get_args, get_data, get_rand_workloads, get_errors
from src.utils import get_per_round_budget_zCDP
from src.syndata import NeuralNetworkGenerator
from src.algo.nondp import IterativeAlgoNonDP
from src.algo import IterAlgoGEM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args(base='nn', iterative='gem', public=True)

data = get_data(args.dataset)

workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQMTorch(data, workloads, device=device)

delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

# load pretrained GEM weights
model_public_save_dir = './save/GEM_Nondp/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset_pub,
                                                                        args.marginal, args.workload,
                                                                        args.workload_seed,
                                                                        args.dim, args.K, args.resample)
G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=None, resample=args.resample)
algo_nondp = IterativeAlgoNonDP(G, args.T,
                                default_dir=model_public_save_dir, verbose=args.verbose, seed=args.test_seed,
                                loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                                max_idxs=args.max_idxs, max_iters=args.max_iters)
algo_nondp.load('best.pkl')

# initialize GEM
model_save_dir = './save/GEM_Pub/{}/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset, args.dataset_pub,
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
algo.G.generator.load_state_dict(algo_nondp.G.generator.state_dict())

true_answers = query_manager.get_answers(data)
algo.fit(true_answers)

syndata = G.get_syndata()
syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)

print(errors)