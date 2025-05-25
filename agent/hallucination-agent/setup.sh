#!/bin/bash

# Create and activate conda environment
conda create -y -n hallucination_slurm python=3.10
conda activate hallucination_slurm

# Load CUDA module - adjust version as needed for your cluster
module load cuda11.8/toolkit/11.8.0
echo "Loaded CUDA module for GPU support"

echo "Setting up hallucination_slurm detection environment for GPU cluster..."

# Install basic dependencies first
pip install certifi urllib3 distro>=1.7.0
pip install packaging>=20.9 pyyaml>=5.1 pytz>=2020.1 six>=1.14.0 jinja2

# Install numpy with the right version and fsspec
pip install "numpy>=1.22.4,<2.0.0"  # Compatible with captum but newer for pandas
pip install "fsspec[http]>=2023.1.0,<=2024.12.0"  # Version compatible with datasets

# Install scientific packages that depend on numpy
pip install scipy matplotlib!=3.6.1,>=3.4
pip install scikit-learn  # Explicitly install scikit-learn

# Install pandas and related visualization
pip install pandas==2.2.3  # Specific version
pip install seaborn==0.13.2  # Specific version

# GPU setup for cluster environment
echo "Setting up GPU environment for cluster..."

# Set visible CUDA devices - if needed for your cluster
# Uncomment and modify the line below if you need to specify certain GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Install CUDA toolkit using conda
conda install -y cudatoolkit=11.8
conda install -y -c conda-forge cudnn=9.0

# Set up library paths for CUDA - important for cluster environments
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX

# Install PyTorch with specific CUDA version for cluster compatibility
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# First install huggingface-hub with compatible version to avoid dependency conflicts
pip install huggingface-hub>=0.24.0

# Install transformers with specific version for compatibility
pip install transformers==4.35.0

# Verify numpy version after transformers installation
python -c "import numpy; print(f'NumPy version after transformers install: {numpy.__version__}')"

# Install dependencies that depend on torch/transformers
pip install captum==0.8.0  # Version that needs numpy<2.0
pip install bert-score==0.3.13
pip install accelerate sentencepiece evaluate rouge-score gputil einops

# Install NLP tools with specific version of spacy to avoid thinc conflict
pip install spacy==3.6.1  # Use older spacy version to avoid thinc 8.3.6 (needs numpy>=2.0)
python -m spacy download en_core_web_sm

# Install selfcheckgpt with compatible version
pip install selfcheckgpt==0.0.9  # Earlier version compatible with our dependencies

# Install datasets with fixed version
pip install datasets==3.5.0

# Verify dependencies after installation
python -c "import huggingface_hub; print(f'huggingface-hub version: {huggingface_hub.__version__}'); import datasets; print(f'datasets version: {datasets.__version__}'); import accelerate; print(f'accelerate version: {accelerate.__version__}')"
