#!/bin/bash
#SBATCH --job-name=transformer2cnn
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

cd $HOME/transformer2cnn
source transformer2cnn/bin/activate
cd dataset/

#python -u get_cutouts.py

python -u vits_finetune.py