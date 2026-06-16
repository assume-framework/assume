#!/bin/bash

# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

#SBATCH --job-name=pp_46   # Name of your job
#SBATCH --output=logs/output_%j.txt   # Output log file (%j = job ID)
#SBATCH --error=logs/error_%j.txt     # Error log file
#SBATCH --time=48:00:00               # Max runtime (hh:mm:ss)
#SBATCH --partition=normal			  # Partition to submit to (adjust as needed)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=64G                     # Memory

# Load modules or activate environment
source /assume-case-py311/bin/activate


# Run your Python script
yes | python examples/examples.py
