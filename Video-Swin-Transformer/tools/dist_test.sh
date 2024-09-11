#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$GPUS --master_port=$PORT \
$(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

# bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2.py 
# checkpoints/swin_base_patch244_window1677_sthv2.pth  3 --eval top_k_accuracy