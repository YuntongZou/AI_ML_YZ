#!/bin/bash
#SBATCH -A p32603                       # Replace with your allocation ID
#SBATCH -p gengpu                       # Specify the GPU partition
#SBATCH --gres=gpu:h100:1               # Request 1 H100 GPU
#SBATCH --constraint=rhel8                # Use the SXM A100 for 80GB GPU memory
#SBATCH -N 1                            # Request 1 node
#SBATCH -n 1                            # Request 1 task
#SBATCH -t 48:00:00                     # Set the maximum runtime for long calculations (48 hours)
#SBATCH --mem=128GB                      # Request 128 GB of CPU memory

# Specify output file paths
#SBATCH -o simulation_output_%j.log     # Standard output log file (%j is the job ID)
#SBATCH -e simulation_error_%j.log # Standard error log file

# Execute the Python script
python YZ_Trasnformer_1M_ES_Single.py



