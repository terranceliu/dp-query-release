from src.syndata.histogram import NormalizedHistogram
from src.algo.nhist_nondp import MultiplicativeWeights
from src.utils.arguments import get_args
from src.qm import KWayMarginalSupportQM
from src.utils.utils_data import get_data, get_rand_workloads
from src.utils.utils_general import get_errors

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
mw = MultiplicativeWeights(G, args.T, default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed)

# get the true answers to our evaluation queries
A_real = query_manager.convert_to_support_distr(data)
true_answers = query_manager.get_answers(A_real)

# run algorithm
mw.fit(true_answers)

# get output of the algorithm
syndata_answers = G.get_answers()

# evaluate error
errors = get_errors(true_answers, syndata_answers)
print(errors)