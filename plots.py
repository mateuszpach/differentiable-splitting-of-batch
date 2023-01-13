import matplotlib.pyplot as plt
import torch
import wandb


def plot_weights_distribution_stacked(weights, epoch):
    labels = list(range(weights[0].size(0)))
    width = 0.9
    fig, ax = plt.subplots()

    y = torch.zeros_like(weights[0], requires_grad=False)
    for i, w in enumerate(weights):
        ax.bar(labels, w.squeeze(), width, bottom=y.squeeze(), label=f'Head {i}')
        y += w

    ax.set_xlabel('Images')
    ax.set_ylabel('Weights')

    wandb.log({"plot_weights_distribution_over_heads": plt}, step=epoch)


def plot_weights_distribution(weights, epoch):
    labels = list(range(weights[0].size(0)))
    width = 0.9

    for i, w in enumerate(weights):
        fig, ax = plt.subplots()
        ax.bar(labels, w.squeeze(), width)

        ax.set_xlabel('Images')
        ax.set_ylabel('Weights')

        wandb.log({f"plot_weights_distribution_head_{i}": plt}, step=epoch)
