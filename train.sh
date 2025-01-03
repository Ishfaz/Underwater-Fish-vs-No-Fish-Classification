#!/bin/bash
#SBATCH --job-name=fish           # Job name
#SBATCH --output=output_torchmean_%j.log    # Output file
#SBATCH --error=errortorchmean_%j.log      # Error file
#SBATCH --partition=GPUQ           # Partition name for GPU access
#SBATCH --gres=gpu:a100:1          # Request 1 A100 GPU with 40GB
#SBATCH --cpus-per-task=4          # Request 4 CPU cores
#SBATCH --mem=40G                  # Request 40 GB of RAM
#SBATCH --time=60:00:00            # Set a time limit of 10 hours

module load CUDA/12.4.0
source ~/miniconda3/bin/activate fish
# Run your Python script    
python /cluster/home/ishfaqab/Thesis/torch_fish.py
