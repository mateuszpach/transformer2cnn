#!/bin/bash
#SBATCH --job-name=transformer2cnn
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=student

cd $HOME/transformer2cnn
source transformer2cnn/bin/activate

python -u run.py