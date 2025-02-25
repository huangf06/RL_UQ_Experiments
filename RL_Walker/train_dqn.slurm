#!/bin/bash
#SBATCH --job-name=walker_train_dqn          # Job name
#SBATCH --output=/home/$USER/RL_UQ_Experiments/results/train_output.txt  # Save stdout
#SBATCH --error=/home/$USER/RL_UQ_Experiments/results/train_error.txt  # Save stderr
#SBATCH --time=1:00:00               # Maximum runtime
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --nodelist=node003

# Ensure script exits if any command fails
set -e  

# Load a clean environment
module purge
module load python/3.9

# Activate virtual environment
source ~/das6_env/bin/activate

# Navigate to script directory
cd ~/RL_UQ_Experiments/RL_Walker/scripts

# Run training
python train_dqn.py
