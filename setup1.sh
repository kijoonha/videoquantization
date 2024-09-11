#!/bin/bash

# Source the .bashrc to load environment variables

source ~/.bashrc

# Activate the conda environment
conda activate swin

# # Load the CUDA module
# module load cuda-11.8

# Change directory to the Video-Swin-Transformer
cd ~/Video-Swin-Transformer

# Print a message to indicate the setup is complete
echo "Environment setup complete. You are now in the Video-Swin-Transformer directory."
