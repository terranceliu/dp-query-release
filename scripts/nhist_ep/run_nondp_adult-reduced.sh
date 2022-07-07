#!/bin/bash

DATASET=adult-reduced

MARGINAL=3
WORKLOAD=35
WORKLOAD_SEED=0

T=1000

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/nondp/run_ep.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--T $T \
--verbose --test_seed 0