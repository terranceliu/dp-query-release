#!/bin/bash

DATASET=acs_CA

MARGINAL=3
WORKLOAD=3200
WORKLOAD_SEED=0

K=1000
DIM=256

T=5000
LOSS_P=2
LR=5e-4
ETA_MIN=1e-7
MAX_IDXS=10000
MAX_ITERS=1

python examples/run/nondp/run_gen_nn.py --sample_by_error --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--K $K --dim $DIM \
--T $T --lr $LR --eta_min $ETA_MIN \
--loss_p $LOSS_P --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--log_freq 1 --sample_by_error --save_best \
--verbose --test_seed 0