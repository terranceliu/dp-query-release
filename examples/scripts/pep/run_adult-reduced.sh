#!/bin/bash

DATASET=adult-reduced

MARGINAL=3
WORKLOAD=35
WORKLOAD_SEED=0

EPSILON=1.0
T=25
ALPHA=0.5

MAX_ITERS=25

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/run_pep.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --T $T --alpha $ALPHA \
--max_iters $MAX_ITERS \
--verbose --test_seed 0