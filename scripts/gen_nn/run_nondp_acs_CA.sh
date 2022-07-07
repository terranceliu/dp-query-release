#!/bin/bash

DATASET=acs_CA

MARGINAL=3
WORKLOAD=3200
WORKLOAD_SEED=0

K=1000
DIM=256

T=10000
LOSS_P=2
LR=5e-4
ETA_MIN=1e-7
MAX_IDXS=10000
MAX_ITERS=1

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/nondp/run_gen_nn.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--K $K --dim $DIM \
--T $T --lr $LR --eta_min $ETA_MIN \
--loss_p $LOSS_P --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--verbose --test_seed 0