# 접속방법
    ssh gpu4
    source ~/.bashrc
    - 쿠다 모듈
      module load cuda-11.8(aiot랩서버)
    - 경로변경 source setup1.sh
    
# 환경설정
    - conda create -n swin python=3.11 ipykernel pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 데이터셋 준비하기
    - zip파일 모두 다운로드 후 unzip한 이후에
    - cat 20bn-something-something-v2-?? | tar -xvzf -

  
# 실행 - inference
   
    python tools/test_quant.py configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py checkpoints/swin_base_patch244_window1677_sthv2.pth  swin_base data/sthv2/videos --eval top_k_accuracy --quant --ptf --lis --quant-method minmax --calib-iter=1000
    bash tools/dist_test.sh  configs/recognition/swin/swin_base_patch244_window1677_sthv2_quant.py checkpoints/swin_base_patch244_window1677_sthv2.pth 4 swin_base data/sthv2/videos --eval top_k_accuracy --quant --ptf --lis --quant-method minmax --calib-iter=1000
    sbatch tools/sbatch_test_quant.sh
