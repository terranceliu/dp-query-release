import torch
from src.qm import KWayMarginalQMTorch
from src.utils import get_args, get_data, get_rand_workloads, get_errors
from src.syndata import NeuralNetworkGenerator
from src.algo.nondp import IterativeAlgoNonDP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args('nn', 'non_dp')

data = get_data(args.dataset)

workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQMTorch(data, workloads, verbose=args.verbose, device=device)
true_answers = query_manager.get_answers(data)

model_save_dir = './save/Gen_NN_NonDP/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset,
                                                                    args.marginal, args.workload, args.workload_seed,
                                                                    args.dim, args.K, args.resample)

G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=None, resample=args.resample)
algo = IterativeAlgoNonDP(G, args.T,
                          default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                          loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                          max_idxs=args.max_idxs, max_iters=args.max_iters,
                          sample_by_error=args.sample_by_error,
                          log_freq=args.log_freq, save_all=args.save_all, save_best=args.save_best,
                          )

algo.fit(true_answers)

syn_answers = G.get_qm_answers()
errors = get_errors(true_answers, syn_answers)
print(errors)