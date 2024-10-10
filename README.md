# 1. 접속방법
    ssh gpu4
    source ~/.bashrc
    - 쿠다 모듈
      module load cuda-11.8(aiot랩서버)
    - 경로변경 source setup1.sh
    
# 2. 환경설정
    - conda create -n swin python=3.11 ipykernel pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 데이터셋 준비하기
    - zip파일 모두 다운로드 후 unzip한 이후에
    - cat 20bn-something-something-v2-?? | tar -xvzf -

  
# 4. 실행 - inference
   
## 1) single gpu
    python tools/test_quant.py configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py       
    checkpoints/swin_base_patch244_window1677_sthv2.pth  swin_base data/sthv2/videos --eval top_k_accuracy --quant --ptf --lis --quant-
    method minmax --calib-iter=1000
## 2) multi gpu
    bash tools/dist_test.sh  configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py 
    checkpoints/swin_base_patch244_window1677_sthv2.pth 4 swin_base data/sthv2/videos --eval top_k_accuracy --quant --ptf --lis --quant-
    method minmax --calib-iter=1000
    
## 3) GSDS 대학원 서버 
     sbatch tools/sbatch_test_quant.sh

# 5. Quantization 적용 파일 경로(모델수정시 수정해야하는 파일)

## 1) calibration, distributed inference 
    videoquantization/Video-Swin-Transformer/tools/test_quant.py
## 2) model backbone
    videoquantization/Video-Swin-Transformer/tools/mmaction/mmaction/models/backbones/swin_transformer.py

# 6. 실험 configuration 파일

## 1) quantization, experiment 관련
    videoquantization/Video-Swin-Transformer/tools/mmcv/utils/config2.py
## 2) swin video model 관련
    videoquantization/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py
