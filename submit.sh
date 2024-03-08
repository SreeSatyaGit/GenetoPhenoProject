#!/bin/bash
#!/usr/bin/env python3

#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=train_cnn
#SBATCH --output=traincnn.out
#SBATCH --mem=30GB
#SBATCH --time=08:00:00
## Load the python interpreter 
module load python/3.7.1
 
#Excute python file.
srun cluster.py