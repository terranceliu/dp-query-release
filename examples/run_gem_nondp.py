import torch

from algorithms.gem import get_args, GEM_Nondp as GEM
from qm import KWayMarginalQM
from utils.utils_data import get_data, get_rand_workloads, get_default_cols

args = get_args()

data = get_data(args.dataset)
data = data.project(get_default_cols(args.dataset))
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQM(data, workloads)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_dir = './save/GEM_Nondp/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset,
                                                           args.marginal, args.workload, args.workload_seed,
                                                           args.dim, args.syndata_size, args.resample)

gem = GEM(query_manager, args.T, device, default_dir=model_save_dir,
          embedding_dim=args.dim, gen_dims=None, K=args.syndata_size,
          loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min, resample=args.resample,
          max_idxs=args.max_idxs, max_iters=args.max_iters, verbose=args.verbose, seed=args.test_seed)

true_answers = query_manager.get_answers(data)
gem.fit(true_answers)