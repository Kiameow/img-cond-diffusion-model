#!/bin/bash
#SBATCH -J zjz
#SBATCH --output=output_%j.log         # Log file with job ID
#SBATCH --error=error_%j.err           # Error log file with job ID
#SBATCH --mail-type=ALL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhang_lumi@foxmail.com 
#SBATCH -A F00120230017
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1

# Load necessary modules
module load cuda12.1/toolkit/12.1.1
# source activate your_conda_env
source ~/.bashrc
cd 
conda env create -f environment.yml
conda activate anomaly_restoration
python3 -m pip install --editable .
bash UPD_study/data/data_preprocessing/prepare_DDR.sh
