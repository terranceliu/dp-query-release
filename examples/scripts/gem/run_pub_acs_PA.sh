#!/bin/bash

DATASET=acs_PA
DATASET_PUB=acs_CA

MARGINAL=3
WORKLOAD=3200
WORKLOAD_SEED=0

EPSILON=1.0
T=200
ALPHA=0.67

K=1000
DIM=256

LOSS_P=2
LR=1e-4
MAX_IDXS=100
MAX_ITERS=100
EMA_WEIGHTS_BETA=0.9

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/public_assisted/run_gem_pub.py --dataset $DATASET --dataset_pub $DATASET_PUB \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --T $T --alpha $ALPHA \
--K $K --dim $DIM \
--loss_p $LOSS_P --lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--ema_weights --ema_weights_beta $EMA_WEIGHTS_BETA \
--verbose --test_seed 0