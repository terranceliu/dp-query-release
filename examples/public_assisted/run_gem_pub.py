import torch

from algorithms.gem import get_args, GEM_Nondp, GEM
from qm import KWayMarginalQM
from utils.utils_data import get_data, get_rand_workloads, get_default_cols
from utils.utils_general import get_errors, get_per_round_budget_zCDP

import pdb

args = get_args()

data = get_data(args.dataset)
data = data.project(get_default_cols(args.dataset))
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQM(data, workloads)

delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load pretrained GEM weights
args.dataset_pub = 'acs_OH'
model_public_save_dir = './save/GEM_Nondp/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset_pub,
                                                           args.marginal, args.workload, args.workload_seed,
                                                           args.dim, args.syndata_size, args.resample)
gem_nondp = GEM_Nondp(query_manager, 10000, data, device, default_dir=model_public_save_dir)
gem_nondp.load('best.pkl')

# initialize GEM
model_save_dir = './save/GEM_Pub/{}/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset, args.dataset_pub,
                                                           args.marginal, args.workload, args.workload_seed,
                                                           args.dim, args.syndata_size, args.resample)
gem = GEM(query_manager, args.T, eps0,
          data, device,
          alpha=args.alpha, default_dir=model_save_dir,
          embedding_dim=args.dim, gen_dim=[args.dim * 2, args.dim * 2],
          batch_size=args.syndata_size, loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min, resample=args.resample,
          max_idxs=args.max_idxs, max_iters=args.max_iters, ema_error_factor=0.5,
          verbose=args.verbose, seed=args.test_seed,
          )
gem.G = gem_nondp.G

true_answers = query_manager.get_answers(data)
gem.fit(true_answers)

syndata = gem.get_syndata()
syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)

print(errors)