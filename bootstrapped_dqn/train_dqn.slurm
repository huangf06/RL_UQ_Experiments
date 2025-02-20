#!/bin/bash
#SBATCH --job-name=train_dqn           # Job name
#SBATCH --output=train_dqn_%j.out      # Standard output log file (%j = job ID)
#SBATCH --error=train_dqn_%j.err       # Standard error log file
#SBATCH --time=12:00:00                # Maximum job run time (hh:mm:ss)
#SBATCH --mem=32G                      # Memory allocation (adjust as needed)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=8               # Request 8 CPU cores for parallel processing

# Load necessary modules (if needed)
module load python/3.9  # Adjust based on the DAS-6 system environment

# Activate your virtual environment (replace with your actual path)
source ~/venv/bin/activate

# Navigate to your project directory (modify if needed)
cd ~/RL_UQ_Experiments

# Run the training script with SLURM-optimized arguments
python train_dqn.py --total_timesteps 500000 --device cuda --logdir logs --save_path models/dqn_cartpole
