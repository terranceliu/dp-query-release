import torch

from qm import KWayMarginalQM
from utils.arguments import get_args
from utils.utils_data import get_data, get_rand_workloads, get_default_cols
from utils.utils_general import get_per_round_budget_zCDP, get_errors, save_results

from algorithms.base.generator import FixedGenerator
from algorithms.rap_softmax import IterAlgoRAPSoftmax

args = get_args(base='fixed', iterative='rap_softmax')

data = get_data(args.dataset)
data = data.project(get_default_cols(args.dataset))
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

query_manager = KWayMarginalQM(data, workloads)

delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T * args.samples_per_round, alpha=args.alpha)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_dir = './save/RAP_Softmax/{}/{}_{}_{}/{}_{}_{}_{}_{}/'.format(args.dataset,
                                                                         args.marginal, args.workload,
                                                                         args.workload_seed,
                                                                         args.epsilon, args.T, args.alpha,
                                                                         args.samples_per_round, args.K)

G = FixedGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed)
algo = IterAlgoRAPSoftmax(G, query_manager, args.T, eps0, device,
                          alpha=args.alpha, default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                          samples_per_round=args.samples_per_round, lr=args.lr, max_iters=args.max_iters, max_idxs=args.max_idxs,
                          )

true_answers = query_manager.get_answers(data)
algo.fit(true_answers)

syndata = algo.get_syndata(args.num_samples)
syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)
print(errors)

save_results("rap_softmax.csv", './results', args, errors)

