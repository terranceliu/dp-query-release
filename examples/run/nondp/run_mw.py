from src.qm import KWayMarginalSupportQM
from src.utils import get_args, get_data, get_rand_workloads
from src.syndata import NormalizedHistogram
from src.algo.nondp import MultiplicativeWeights

args = get_args(base='nhist', iterative='non_dp')

# load dataset (using csv filename)
data = get_data(args.dataset)

# define our (workloads of) evaluation queries
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

# the query manager has methods we use to evaluate queries on an input dataset
query_manager = KWayMarginalSupportQM(data, workloads)

# instantiate class for our algorithm
model_save_dir = './save/MW/{}/{}_{}_{}/{}/'.format(args.dataset, args.marginal, args.workload,
                                                    args.workload_seed, args.T)
G = NormalizedHistogram(query_manager)
algo = MultiplicativeWeights(G, args.T, default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed)

# get the true answers to our evaluation queries
A_real = query_manager.convert_to_support_distr(data)
true_answers = query_manager.get_answers(A_real)

# run algorithm
algo.fit(true_answers)