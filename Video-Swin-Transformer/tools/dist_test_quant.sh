#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
MODEL=$3
DATA=$4
GPUS=$5
PORT=${PORT:-29500}
# PORT=${PORT:-36583}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

# CUDA_VISIBLE_DEVICES=0,1  \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
$(dirname "$0")/test_quant.py $CONFIG $CHECKPOINT $MODEL $DATA --launcher pytorch ${@:6}

# bash tools/dist_test_quant.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py checkpoints/swin_base_patch244_window1677_sthv2.pth swin_base data/sthv2/videos 2 --eval top_k_accuracy --quant --ptf --lis --quant-method minmax --calib-iter=1000