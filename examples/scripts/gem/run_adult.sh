#!/bin/bash

DATASET=adult

MARGINAL=3
WORKLOAD=286
WORKLOAD_SEED=0

EPSILON=1.0
T=50
ALPHA=0.67

K=1000
DIM=256

LOSS_P=2
LR=1e-4
MAX_IDXS=100
MAX_ITERS=100
EMA_WEIGHTS_BETA=0.9

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/run_gem.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--epsilon $EPSILON --T $T --alpha $ALPHA \
--K $K --dim $DIM \
--loss_p $LOSS_P --lr $LR --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--ema_weights --ema_weights_beta $EMA_WEIGHTS_BETA \
--verbose --test_seed 0