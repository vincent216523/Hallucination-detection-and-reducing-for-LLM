#!/bin/bash

#SBATCH -J hallucination
#SBATCH -t 12:00:00
#SBATCH --mail-user=XXX@connect.ust.hk
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -p normal
#SBATCH --nodes=1 --gpus-per-node=2
#SBATCH --account=XXX
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Setup runtime environment - the msbd5002 folder exists in the parent directory
#cd msbd5002 || { echo "Failed to change to msbd5002 directory"; exit 1; }
#echo "Current directory: $(pwd)"

# Check if conda is available
which conda || { echo "Conda not found in PATH"; exit 1; }

# Load Anaconda module if needed
# module load Anaconda3

# List conda environments to verify
conda env list

# Activate existing environment
echo "Activating conda environment hallucination_slurm"
source activate hallucination_slurm || { echo "Failed to activate conda environment"; exit 1; }

# Print environment info for debugging
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')" || echo "PyTorch not installed"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')" || echo "CUDA check failed"
echo "GPU devices: $(python -c 'import torch; print(torch.cuda.device_count())')" || echo "GPU check failed"

# List the contents of the current directory to verify files are present
echo "Files in current directory:"
ls -la

# Set working directory to current directory for PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Check if generate.py exists
if [ ! -f "generate.py" ]; then
    echo "ERROR: generate.py file not found in current directory!"
    exit 1
fi

# Run with Python verbose mode to see import errors
echo "Starting generate.py..."
python -v generate.py --start 1500 --end 2500 --n 3 --sensitivity 0.05 || {
    echo "Script execution failed with exit code $?"
    echo "Checking for potential import issues:"
    python -c "import sys; print('Python path:', sys.path)"
    python -c "import torch; print('Torch location:', torch.__file__)" || echo "Failed to import torch"
    python -c "from datasets import load_dataset; print('datasets library available')" || echo "Failed to import datasets"
    exit 1
}

echo "Job completed at $(date)"
