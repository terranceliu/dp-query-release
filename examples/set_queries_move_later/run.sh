#!/bin/bash

DATASET=ppmf_42003140100

K=1000

T=1000
LOSS_P=2
LR=1e-1
MAX_IDXS=1000
MAX_ITERS=1

python examples/set_queries_move_later/run_gen_fixed.py --dataset $DATASET \
--K $K \
--T $T --lr $LR \
--loss_p $LOSS_P --max_idxs $MAX_IDXS --max_iters $MAX_ITERS \
--verbose --test_seed 0 --log_freq 1 --sample_by_error