#!/bin/bash
set -e

ENV_NAME="cuda_test"
CUDA_VERSION="12.1"  # Compatible with CUDA 12.2 runtime reported by nvidia-smi

echo "Creating conda environment with CUDA support..."
echo "Environment: $ENV_NAME"
echo "CUDA Version: $CUDA_VERSION"

# make conda available on HPC
module load lang/Anaconda3

# enable internet access on HPC
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment with Python 3.12
conda create -n $ENV_NAME python=3.12 -y

# Activate environment
source activate $ENV_NAME

echo "Installing CUDA toolkit via conda..."
# Install CUDA development tools via conda (no system modules needed)
# conda install -c conda-forge cudatoolkit-dev=$CUDA_VERSION -y # newest avail. version: 11.7.0
conda install -c nvidia cuda-toolkit=$CUDA_VERSION -y


echo "Installing PyTorch with CUDA support..."
# Install PyTorch with matching CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y

echo "Installing additional tools..."
# Install build tools
conda install -c conda-forge gcc_linux-64 gxx_linux-64 cmake ninja -y

# Install Python packages
pip install numpy cython

echo "Environment created successfully!"
echo "To activate: conda activate $ENV_NAME"
