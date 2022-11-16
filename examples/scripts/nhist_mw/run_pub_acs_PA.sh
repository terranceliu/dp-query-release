#!/bin/bash

DATASET=acs_PA
DATASET_PUB=acs_OH

MARGINAL=3
WORKLOAD=3200
WORKLOAD_SEED=0

EPSILON=1.0
T=500
ALPHA=0.5

python examples/run/public_assisted/run_pmw_pub.py --dataset $DATASET --dataset_pub $DATASET_PUB \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --T $T --alpha $ALPHA \
--recycle \
--verbose --test_seed 0