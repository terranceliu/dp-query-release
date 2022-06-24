import torch

from qm import KWayMarginalQMTorch
from utils.arguments import get_args
from utils.utils_data import get_data, get_rand_workloads

from algorithms.syndata.generator import NeuralNetworkGenerator
from algorithms.non_dp import IterativeAlgoNonDP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args('nn', 'non_dp')

data = get_data(args.dataset)

workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQMTorch(data, workloads, device=device)

model_save_dir = './save/GEM_Nondp/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset,
                                                                 args.marginal, args.workload, args.workload_seed,
                                                                 args.dim, args.K, args.resample)

G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=None, resample=args.resample)
algo = IterativeAlgoNonDP(G, query_manager, args.T, device,
                          default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                          loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                          max_idxs=args.max_idxs, max_iters=args.max_iters)

true_answers = query_manager.get_answers(data)
algo.fit(true_answers)