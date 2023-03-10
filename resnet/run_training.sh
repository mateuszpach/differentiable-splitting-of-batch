#!/bin/bash
#SBATCH --job-name=dsob
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

export MNIST_PATH=/shared/sets/datasets
export CIFAR10_PATH=/shared/sets/datasets/cifar10
export CIFAR100_PATH=/shared/sets/datasets/cifar100
export TINYIMAGENET_PATH=/shared/sets/datasets/tiny-imagenet-200

export WANDB_ENTITY="mateuszpach"
export WANDB_PROJECT="eet"

cd $HOME/differentiable-splitting-of-batch/resnet

source activate differentiable-splitting-of-batch-cuda


ARGS="--exp_id 71 \
      --model_class resnet18_4heads \
      --dataset cifar10 \
      --epochs 5 \
      --batch_size 512 \
      --loss_type ce \
      --loss_args {\"label_smoothing\":0.1} \
      --optimizer_class sgd \
      --optimizer_args {\"lr\":0.5,\"momentum\":0.9,\"weight_decay\":0.0005} \
      --scheduler_class step_lr \
      --scheduler_args {\"step_size\":10}"
ARGS2="--base_model_state_path runs/cifar10_resnet18_4heads_4PI7IXEN_1/state.pth \
       --eet_loss_factor 1 \
       --expected_exit_time 2.48"
python -u train_expected_exit_time.py $ARGS $ARGS2
