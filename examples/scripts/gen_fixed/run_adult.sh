#!/bin/bash

DATASET=adult

MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0

EPSILON=1.0
T=25
ALPHA=0.5

K=1000
SAMPLES_PER_ROUND=2

LR=1e-1
MAX_IDXS=10000
MAX_ITERS=1000

python examples/run/run_rap.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --alpha $ALPHA --T $T \
--K $K --samples_per_round $SAMPLES_PER_ROUND \
--lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--verbose --test_seed 0