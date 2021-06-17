#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1

module load 2020
module load Python
python3 main.py
