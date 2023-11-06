#!/bin/bash

#SBATCH -A khangroup_gpu 
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --job-name=IR_ABS
#SBATCH --time=2-6:0:0 
#SBATCH --mem=64G 
#SBATCH -o %x-%j.out

python3 src/PRMDIR.py --dataset ir 
