#!/bin/bash

DATASET=adult

MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0

EPSILON=1.0
T=75

DIM=256
SYNDATA_SIZE=1000

LOSS_P=2
MAX_IDXS=10000
MAX_ITERS=100
ALPHA=0.5
EMA_WEIGHTS_BETA=0.5

python examples/run_gem.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--syndata $SYNDATA_SIZE --dim $DIM \
--epsilon $EPSILON --alpha $ALPHA --T $T \
--loss_p $LOSS_P --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--ema_weights --ema_weights_beta $EMA_WEIGHTS_BETA \
--verbose --test_seed 0