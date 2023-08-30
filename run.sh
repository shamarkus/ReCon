#!/bin/bash

#SBATCH -A khangroup_gpu 
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --job-name=STGCN
#SBATCH --time=2-6:0:0 
#SBATCH --mem=64G 
#SBATCH -o %x-%j.out


python3 src/baseline.py 
