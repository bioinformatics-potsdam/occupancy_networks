#!/bin/bash
set -e

ENV_NAME="minimal-onet-py310"
CUDA_VERSION="11.8"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME with CUDA $CUDA_VERSION"

# make conda available on HPC
module load lang/Anaconda3

# enable internet access on HPC
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source activate $ENV_NAME

# Install CUDA toolkit
echo "Installing CUDA toolkit..."
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=$CUDA_VERSION cuda-nvcc=$CUDA_VERSION -y

# Install PyTorch
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y

# Install COMPATIBLE build tools - GCC 11 for CUDA 11.8 compatibility
echo "Installing build tools..."
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake ninja -y

# Install Python packages
echo "Installing Python dependencies..."
conda install cython pykdtree -y

pip install numpy scipy h5py Pillow scikit-image imageio matplotlib pandas \
           PyYAML tensorboard tqdm trimesh plyfile pytest PyMCubes

# Set up environment variables
echo "Setting up CUDA environment variables..."
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh << 'EOF'
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="7.0"
# Ensure we use the conda GCC
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/cuda_env.sh << 'EOF'
unset CUDA_HOME CUDA_ROOT CUDA_PATH TORCH_CUDA_ARCH_LIST CC CXX
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/cuda_env.sh

# Reactivate to load environment variables
conda deactivate
source activate $ENV_NAME

# Quick verification
echo ""
echo "Verification:"
echo "- NVCC: $(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')"
echo "- PyTorch: $(python -c "import torch; print(torch.version.cuda)")"
echo "- GCC: $(gcc --version | head -1)"
echo "- Cython: $(python -c "import Cython; print(Cython.__version__)")"
echo ""
echo "✓ Environment ready!"
echo "To use: conda activate $ENV_NAME"
