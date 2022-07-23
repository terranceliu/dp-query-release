#!/bin/bash

DATASET=adult-reduced

MARGINAL=3
WORKLOAD=35
WORKLOAD_SEED=0

EPSILON=1.0
T=100
ALPHA=0.5

python examples/run/run_mwem.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --T $T --alpha $ALPHA \
--recycle \
--verbose --test_seed 0