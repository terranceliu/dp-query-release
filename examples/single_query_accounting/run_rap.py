import torch

from rap import RAP, get_args
from qm import KWayMarginalQM
from utils.utils_data import get_data, get_rand_workloads
from utils.utils_general import get_errors, get_per_round_budget_zCDP

args = get_args()

data = get_data(args.dataset)
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQM(data, workloads)

delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_dir = './save/RAP/{}/{}_{}_{}/{}_{}_{}_{}_{}/'.format(args.dataset,
                                                                 args.marginal, args.workload, args.workload_seed,
                                                                 args.epsilon, args.T, args.alpha, args.K, args.n)
rap = RAP(query_manager, args.T, eps0,
          data, device,
          alpha=args.alpha, default_dir=model_save_dir,
          n=args.n, K=args.K,
          softmax=args.softmax,
          lr=args.lr, max_iters=args.max_iters, max_idxs=args.max_idxs,
          verbose=args.verbose, seed=args.test_seed,
          )

true_answers = query_manager.get_answers(data)
rap.fit(true_answers)

syndata_answers = rap.get_answers()
errors = get_errors(true_answers, syndata_answers)

print(errors)



