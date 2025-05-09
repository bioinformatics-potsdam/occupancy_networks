# Dockerfile for Occupancy Networks (Ubuntu 18.04, Python 3.6, CPU)
# Uses pre-modified setup.py, im2mesh/config.py and im2mesh/checkpoints.py via COPY

ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION}

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Build environment for Occupancy Networks (CPU version) with specific Python dependencies and fixes for compilation and import errors."

# set environment variables to avoid interactive prompts and ensure UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# install system dependencies
# - python3.6 and headers for building extensions
# - python3-pip for package installation
# - build-essential for C/C++ compilation
# - git for cloning the repository
# - libomp-dev needed by pykdtree
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.6 \
        python3.6-dev \
        python3-pip \
        build-essential \
        git \
        libomp-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# set up python3.6 as the default python3 and pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# upgrade pip and install basic build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir wheel setuptools

# install Python dependencies
RUN python3 -m pip install --no-cache-dir torch==1.0.1 torchvision==0.2.1
RUN python3 -m pip install --no-cache-dir numpy==1.15.4

# --- WORKAROUND: Copy NumPy headers to global include path ---
# This helps compilation steps find "numpy/arrayobject.h" even if not explicitly included
RUN cp -r /usr/local/lib/python3.6/dist-packages/numpy/core/include/numpy /usr/local/include/numpy

# install remaining dependencies
RUN python3 -m pip install --no-cache-dir \
        # Core scientific stack & compilation needs
        scipy==1.1.0 \
        cython==0.29.2 \
        h5py==2.9.0 \
        # Skipped: pyembree==0.1.4 (requires specific setup) \
        # Image/Plotting Stack
        Pillow==5.3.0 \
        scikit-image==0.14.1 \
        imageio==2.4.1 \
        matplotlib==3.0.3 \
        # Other Dependencies
        pandas==0.23.4 \
        PyYAML==3.13 \
        tensorboardX==1.4 \
        tqdm==4.28.1 \
        trimesh==2.37.7 \
        plyfile==0.7 \
        pytest==4.0.2


# add NVIDIA CUDA repository
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb

# install CUDA toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-10-1 \
    && rm -rf /var/lib/apt/lists/*

# set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


# set final working directory / copy repo into container
WORKDIR /app/occupancy_networks
COPY eval_meshes.py eval.py generate.py setup.py train.py /app/occupancy_networks/
COPY configs /app/occupancy_networks/configs
COPY demo /app/occupancy_networks/demo
COPY external /app/occupancy_networks/external
COPY im2mesh /app/occupancy_networks/im2mesh
COPY img /app/occupancy_networks/img
COPY scripts /app/occupancy_networks/scripts

# compile Cython/C++ extensions
RUN python3 setup.py build_ext --inplace

# run bash when container starts
CMD ["/bin/bash"]
