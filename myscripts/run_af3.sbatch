#!/bin/bash
#SBATCH --job-name=af3
#SBATCH --output=run.out
#SBATCH --error=run.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=oxygen
#SBATCH --gres=gpu:1

# Activate conda environment
source /home/hehuang/miniconda3/etc/profile.d/conda.sh
conda activate af3
af3_dir=/home/hehuang/Tools/alphafold3

# Source and run the alphafold script
bash ${af3_dir}/run_alphafold.sh
