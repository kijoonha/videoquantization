#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}


torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# bash tools/dist_train.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2.py \
#     3 --test-last --validate --cfg-options \
#     model.backbone.pretrained=openaiclip \
#     work_dir=sthsthv2_patch224_window1677/20240718