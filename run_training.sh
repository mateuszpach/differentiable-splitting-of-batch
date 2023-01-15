#!/bin/bash
#SBATCH --job-name=dsob
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

export MNIST_PATH=/shared/sets/datasets
export CIFAR10_PATH=/shared/sets/datasets/cifar10
export CIFAR100_PATH=/shared/sets/datasets/cifar100
export TINYIMAGENET_PATH=/shared/sets/datasets/tiny-imagenet-200

export WANDB_ENTITY="mateuszpach"
export WANDB_PROJECT="dsob2"

cd $HOME/differentiable-splitting-of-batch

source activate differentiable-splitting-of-batch-cuda

ARGS="--exp_id 16 \
      --model_class resnet18_4heads \
      --dataset cifar10 \
      --epochs 50 \
      --batch_size 128 \
      --loss_type ce \
      --loss_args {\"label_smoothing\":0.1} \
      --optimizer_class sgd \
      --optimizer_args {\"lr\":0.01,\"momentum\":0.9,\"weight_decay\":0.0005} \
      --scheduler_class cosine \
      --scheduler_args {}"
python -u train.py $ARGS

ARGS="--exp_id 17 \
      --model_class resnet18_4heads_dsob \
      --dataset cifar10 \
      --epochs 50 \
      --batch_size 128 \
      --loss_type ce \
      --loss_args {\"label_smoothing\":0.1} \
      --optimizer_class sgd \
      --optimizer_args {\"lr\":0.01,\"momentum\":0.9,\"weight_decay\":0.0005} \
      --scheduler_class cosine \
      --scheduler_args {}"
python -u train.py $ARGS
#
#ARGS="--exp_id 32 \
#      --model_class resnet18_4heads_dsob \
#      --dataset cifar10 \
#      --epochs 50 \
#      --batch_size 128 \
#      --loss_type ce \
#      --loss_args {\"label_smoothing\":0.1} \
#      --optimizer_class sgd \
#      --optimizer_args {\"lr\":0.01,\"momentum\":0.9,\"weight_decay\":0.0005} \
#      --scheduler_class cosine \
#      --scheduler_args {}"
#
#python -u train.py $ARGS