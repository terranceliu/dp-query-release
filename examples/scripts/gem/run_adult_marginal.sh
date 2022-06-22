#!/bin/bash

DATASET=adult

MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0

EPSILON=1.0
T=5
ALPHA=0.5

K=1000
DIM=256

LOSS_P=2
LR=1e-4
MAX_IDXS=10000
MAX_ITERS=100

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/marginal_trick/run_gem.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --T $T --alpha $ALPHA \
--K $K --dim $DIM \
--loss_p $LOSS_P --lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--verbose --test_seed 0