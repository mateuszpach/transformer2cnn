#!/bin/bash
#SBATCH --job-name=transformer2cnn
#SBATCH --qos=test
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

cd $HOME/transformer2cnn
python3 -m venv transformer2cnn
source transformer2cnn/bin/activate
pip install torch Pillow pytorch-lightning transformers numpy torchvision tensorboard tensorboardX wandb matplotlib scikit-learn