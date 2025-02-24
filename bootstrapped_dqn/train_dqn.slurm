#!/bin/bash
#SBATCH --job-name=train_dqn          # Job name
#SBATCH --output=train_dqn_%j.out     # Standard output log file
#SBATCH --error=train_dqn_%j.err      # Standard error log file
#SBATCH --time=1:00:00               # Maximum runtime
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=4              # Reduce CPU usage
#SBATCH --nodelist=node003             # Use an available node

# Load Python 3.9 module (adjust if needed)
module load python/3.9

# Activate virtual environment
source ~/das6_env/bin/activate

# Navigate to your project directory
cd ~/RL_UQ_Experiments

# Run DQN training
python train_dqn.py --total_timesteps 500000 --device cuda --logdir logs --save_path models/dqn_cartpole
