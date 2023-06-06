#!/bin/bash
#SBATCH --job-name=transformer2cnn
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

cd $HOME/transformer2cnn
source transformer2cnn/bin/activate

python -u embeddings_visualization.py --dataset CUB200 --model dinovit --checkpoint dataset/final.ckpt
python -u embeddings_visualization.py --dataset CIFAR10 --model dinovit --checkpoint dataset/final.ckpt
python -u embeddings_visualization.py --dataset CUB200 --model dino2resnet --checkpoint final.ckpt
python -u embeddings_visualization.py --dataset CIFAR10 --model dino2resnet --checkpoint final.ckpt
