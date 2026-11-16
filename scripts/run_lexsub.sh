#!/bin/bash
#SBATCH --job-name=lexsub_aug
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/others/15407/khasi_mt/logs/lexsub_%j.log
#SBATCH --error=/home/others/15407/khasi_mt/logs/lexsub_%j.err

export PATH=/home/others/15407/.conda/envs/anyway_mt/bin:$PATH

python /home/others/15407/khasi_mt/scripts/lexical_substitution.py
