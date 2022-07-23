from src.qm import KWayMarginalSupportQM
from src.utils import get_args, get_data, get_rand_workloads, get_errors, save_results
from src.utils import get_per_round_budget_zCDP
from src.syndata import NormalizedHistogram
from src.algo import MWEMSingle as MWEM

args = get_args(base='nhist', iterative='mwem')

# load dataset (using csv filename)
data = get_data(args.dataset)

# define our (workloads of) evaluation queries
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

# the query manager has methods we use to evaluate queries on an input dataset
query_manager = KWayMarginalSupportQM(data, workloads)

# defines differential privacy parameters (zCDP)
delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

# instantiate class for our algorithm
model_save_dir = './save/MWEM/{}/{}_{}_{}/{}_{}_{}_{}/'.format(args.dataset,
                                                               args.marginal, args.workload, args.workload_seed,
                                                               args.epsilon, args.T, args.alpha, args.recycle)

G = NormalizedHistogram(query_manager)
algo = MWEM(G, args.T, eps0,
            alpha=args.alpha, recycle_queries=args.recycle,
            default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
            )

# get the true answers to our evaluation queries
A_real = query_manager.convert_to_support_distr(data)
true_answers = query_manager.get_answers(A_real)

# run algorithm
algo.fit(true_answers)

# get output of the algorithm
syndata_answers = G.get_answers()

# evaluate error
errors = get_errors(true_answers, syndata_answers)
print(errors)