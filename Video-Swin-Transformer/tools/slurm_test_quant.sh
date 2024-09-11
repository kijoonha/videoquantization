#!/usr/bin/env bash

set -x

# SLURM 파티션 이름과 작업 이름을 인자로 받습니다.
PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
MODEL=$5
DATA=$6
GPUS=$7
PY_ARGS=${@:8}  # 여덟 번째 인수 이후의 모든 인수를 캡처

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

# PYTHONPATH를 설정하고 srun을 사용하여 작업 실행
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test_quant.py ${CONFIG} ${CHECKPOINT} ${MODEL} ${DATA} --launcher="slurm" ${PY_ARGS}

#명령어
#srun bash tools/slurm_test_quant.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py 
# checkpoints/swin_base_patch244_window1677_sthv2.pth swin_base data/sthv2/videos 2 --eval top_k_accuracy --quant --ptf --lis --quant-method minmax --calib-iter=1000bash tools/dist_test_slurm.sh P2 SwinQuant 
# configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py checkpoints/swin_base_patch244_window1677_sthv2.pth swin_base data/sthv2/videos 2 --eval top_k_accuracy --quant --ptf 
# --lis --quant-method minmax --calib-iter=1000