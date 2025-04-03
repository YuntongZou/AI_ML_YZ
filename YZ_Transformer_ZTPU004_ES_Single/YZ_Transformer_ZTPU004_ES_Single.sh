#!/bin/bash
#SBATCH -A p32603                       # Replace with your allocation ID
#SBATCH -p gengpu                       # Specify the GPU partition
#SBATCH --gres=gpu:a100:1               # Request 1 A100 GPU
#SBATCH --constraint=sxm                # Use the SXM A100 for 80GB GPU memory
#SBATCH -N 1                            # Request 1 node
#SBATCH -n 1                            # Request 1 task
#SBATCH -t 48:00:00                     # Set the maximum runtime for long calculations (48 hours)
#SBATCH --mem=32G                      # Request 128 GB of CPU memory

# Specify output file paths
#SBATCH -o simulation_output_%j.log     # Standard output log file (%j is the job ID)
#SBATCH -e simulation_error_%j.log	# Standard error log file


# Execute the Python script
python tYZ_Transformer_ZTPU004_ES_Single.py



