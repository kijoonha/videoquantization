#!/bin/bash

#SBATCH --job-name=SwinQ                               # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:4                            # Using 4 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=40000MB                         # Using 20GB CPU Memory
#SBATCH --partition=P2                        # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor
#SBATCH --output=/home/s3/joonhaki/Video-Swin-Transformer/work_dirs/slurm-%j.out     # 표준 출력 파일



source ${HOME}/.bashrc
source ${HOME}/miniconda3/bin/activate
conda activate swin

srun bash tools/dist_test_quant.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py checkpoints/swin_base_patch244_window1677_sthv2.pth swin_base data/sthv2/videos 4 --eval top_k_accuracy --quant --ptf --lis --quant-method minmax --calib-iter=1000
