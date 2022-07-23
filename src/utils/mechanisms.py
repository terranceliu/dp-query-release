import torch
import numpy as np
from scipy.special import softmax

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

    if torch.is_tensor(scores):
        EM_dist = torch.softmax(2 * eps0 * scores / (2 * sensitivity), dim=-1)
        cumulative_dist = EM_dist.cumsum(-1)
        max_query_idx = torch.searchsorted(cumulative_dist, torch.rand(1, device=cumulative_dist.device))
    else:
        EM_dist = softmax(2 * eps0 * scores / (2 * sensitivity))
        cumulative_dist = np.cumsum(EM_dist)
        max_query_idx = np.searchsorted(cumulative_dist, np.random.rand())
    return max_query_idx

def report_noisy_max(input, eps0, sensitivity):
    pass


# MEASUREMENT mechanisms
def gaussian_mech(true_val, eps0, sensitivity):
    try:
        size = true_val.shape
    except:
        size = 1

    if torch.is_tensor(true_val):
        noise = torch.normal(0, sensitivity / eps0, size=size, device=true_val.device)
    else:
        noise = np.random.normal(loc=0, scale=sensitivity / eps0, size=size)
    return true_val + noise

def laplace_mech(input, eps0, sensitivity):
    pass