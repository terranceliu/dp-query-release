#!/bin/bash

DATASET=acs_CA

MARGINAL=3
WORKLOAD=3200
WORKLOAD_SEED=0

DIM=256
SYNDATA_SIZE=1000

LOSS_P=2
MAX_IDXS=10000
MAX_ITERS=1

export PYTHONPATH="${PYTHONPATH}:${PWD}$"

python examples/run_gem_nondp.py --dataset $DATASET \
--marginal $MARGINAL --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
--syndata $SYNDATA_SIZE --dim $DIM \
--loss_p $LOSS_P --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--verbose --test_seed 0 --T 10000 --lr 5e-4 --eta_min 1e-7