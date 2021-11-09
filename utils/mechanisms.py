import torch
import numpy as np

"""
Privacy mechanisms
"""

def sample(dist):
    cumulative_dist = np.cumsum(dist)
    r = np.random.rand()
    return np.searchsorted(cumulative_dist, r)

# SELECTION mechanisms
# get max error query /w exponential mechanism (https://arxiv.org/pdf/2004.07223.pdf Lemma 3.2)
def exponential_mech(scores, eps0, sensitivity):
    if eps0 == 0: # for running iterative algo with EM (sample random query each round)
        idxs = np.arange(len(scores))
        idxs = idxs[scores != -np.infty]
        return np.random.choice(idxs)

    EM_dist = np.exp(2 * eps0 * scores / (2 * sensitivity), dtype=np.float128)
    EM_dist = EM_dist / EM_dist.sum()
    max_query_idx = sample(EM_dist)
    return max_query_idx

def report_noisy_max(input, eps0, sensitivity):
    pass


# MEASUREMENT mechanisms
def gaussian_mech(true_val, eps0, sensitivity):
    try:
        size = true_val.shape
    except:
        size = 1

    noise = np.random.normal(loc=0, scale=sensitivity / eps0, size=size)
    if isinstance(true_val, torch.Tensor):
        noise = torch.tensor(noise)
    return true_val + noise

def laplace_mech(input, eps0, sensitivity):
    pass