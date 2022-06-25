from algorithms.public_assisted.pmw_pub import PMWPub
from utils.arguments import get_args
from qm import KWayMarginalQM, KWayMarginalSupportQMPublic
from utils.utils_data import get_data, get_rand_workloads, get_default_cols
from utils.utils_general import get_errors, get_per_round_budget_zCDP

args = get_args(base='approx', iterative='mwem', public=True)

# load dataset (using csv filename)
data = get_data(args.dataset)
data = data.project(get_default_cols(args.dataset))
data_public = get_data(args.dataset_pub)
data_public = data_public.project(get_default_cols(args.data_public))

# define our (workloads of) evaluation queries
workloads = get_rand_workloads(data, args.workload, args.marginal, seed=args.workload_seed)

# the query manager has methods we use to evaluate queries on an input dataset
query_manager_public = KWayMarginalSupportQMPublic(data_public, workloads)

# override support with the support of the public dataset

# defines differential privacy parameters (zCDP)
delta = 1.0 / len(data) ** 2
eps0, rho = get_per_round_budget_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)

# instantiate class for our algorithm
model_save_dir = './save/PMWPub/{}/{}/{}_{}_{}/{}_{}_{}_{}/'.format(args.dataset, args.dataset_pub,
                                                                    args.marginal, args.workload, args.workload_seed,
                                                                    args.epsilon, args.T, args.alpha, args.recycle)
mwem = PMWPub(query_manager_public, args.T, eps0,
              alpha=args.alpha, default_dir=model_save_dir,
              recycle_queries=args.recycle,
              verbose=args.verbose, seed=args.test_seed,
              )

# get the true answers to our evaluation queries
query_manager = KWayMarginalQM(data, workloads)
true_answers = query_manager.get_answers(data)

# run algorithm
mwem.fit(true_answers)

# get output of the algorithm
syndata = mwem.get_syndata()
syndata_answers = query_manager_public.get_answers(syndata)

# evaluate error
errors = get_errors(true_answers, syndata_answers)
print(errors)