#!/bin/bash
#SBATCH --job-name=dsob-bert
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

source activate differentiable-splitting-of-batch-cuda

python -u run_glue_script.py --config glue_cfg.json