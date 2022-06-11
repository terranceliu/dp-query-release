#!/bin/bash

DATASET=acs_PA

MARGINAL=3
WORKLOAD=3200
WORKLOAD_SEED=0

EPSILON=1.0
T=200

DIM=256
SYNDATA_SIZE=1000

LOSS_P=2
MAX_IDXS=100
MAX_ITERS=100
ALPHA=0.67
EMA_WEIGHTS_BETA=0.5

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/public_assisted/run_gem_pub.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--syndata $SYNDATA_SIZE --dim $DIM \
--epsilon $EPSILON --alpha $ALPHA --T $T \
--loss_p $LOSS_P --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--ema_weights --ema_weights_beta $EMA_WEIGHTS_BETA \
--verbose --test_seed 0