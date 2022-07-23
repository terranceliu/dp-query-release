import torch

from src.qm import KWayMarginalQMTorch
from src.utils.arguments import get_args
from src.utils.utils_data import get_data, get_rand_workloads

from src.syndata.generator import NeuralNetworkGenerator
from src.algo.gen_nondp import IterativeAlgoNonDP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args('nn', 'non_dp')

data = get_data(args.dataset)

workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQMTorch(data, workloads, device=device)

model_save_dir = './save/Gen_NN_NonDP/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset,
                                                                    args.marginal, args.workload, args.workload_seed,
                                                                    args.dim, args.K, args.resample)

G = NeuralNetworkGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed,
                           embedding_dim=args.dim, gen_dims=None, resample=args.resample)
algo = IterativeAlgoNonDP(G, args.T,
                          default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                          loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                          max_idxs=args.max_idxs, max_iters=args.max_iters)

true_answers = query_manager.get_answers(data)
algo.fit(true_answers)