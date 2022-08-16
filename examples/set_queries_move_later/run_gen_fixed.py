import torch
import pickle
from src.qm import KWayMarginalSetQMTorch
from src.utils import get_args, get_data, get_errors
from src.syndata import FixedGenerator
from src.algo.nondp import IterativeAlgoNonDP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args('nn', 'non_dp')

data = get_data(args.dataset)

queries_path = './datasets/queries/{}-set.pkl'.format(args.dataset)
with open(queries_path, 'rb') as handle:
    queries = pickle.load(handle)

query_manager = KWayMarginalSetQMTorch(data, queries, verbose=args.verbose, device=device)
true_answers = query_manager.get_answers(data)

model_save_dir = './save/Gen_NN_NonDP/{}/{}_{}_{}/{}_{}_{}/'.format(args.dataset,
                                                                    args.marginal, args.workload, args.workload_seed,
                                                                    args.dim, args.K, args.resample)

G = FixedGenerator(query_manager, K=args.K, device=device, init_seed=args.test_seed)
algo = IterativeAlgoNonDP(G, args.T,
                          default_dir=model_save_dir, verbose=args.verbose, seed=args.test_seed,
                          loss_p=args.loss_p, lr=args.lr, eta_min=args.eta_min,
                          max_idxs=args.max_idxs, max_iters=args.max_iters,
                          sample_by_error=args.sample_by_error,
                          log_freq=args.log_freq, save_all=args.save_all, save_best=args.save_best,
                          )

algo.fit(true_answers)

syn_answers = G.get_qm_answers()
errors = get_errors(true_answers, syn_answers)
print(errors)

syndata = G.get_syndata(args.num_samples)
syndata_answers = query_manager.get_answers(syndata)
errors = get_errors(true_answers, syndata_answers)
print(errors)

import pdb
pdb.set_trace()