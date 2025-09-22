#!/bin/bash
#SBATCH --job-name=learn_bat         # Name of your job
#SBATCH --output=logs/output_%j.txt   # Output log file (%j = job ID)
#SBATCH --error=logs/error_%j.txt     # Error log file
#SBATCH --time=48:00:00               # Max runtime (hh:mm:ss)
#SBATCH --partition=normal
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=64G                     # Memory

# Load modules or activate environment
source ~/venvs/assume-framework/bin/activate


# Run your Python script
yes | python examples/examples.py
