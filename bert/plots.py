import matplotlib.pyplot as plt
import numpy as np
import wandb


def plot_weights_distribution_stacked(weights):
    labels = list(range(weights[0].shape[0]))
    width = 0.9
    fig, ax = plt.subplots()

    y = np.zeros_like(weights[0])
    for i, w in enumerate(weights):
        ax.bar(labels, w.squeeze(), width, bottom=y.squeeze(), label=f'Head {i}')
        y += w

    ax.set_xlabel('Samples')
    ax.set_ylabel('Weights')

    wandb.log({"plot_weights_distribution_over_heads": plt})


def plot_weights_distribution(weights):
    labels = list(range(weights[0].shape[0]))
    width = 0.9

    for i, w in enumerate(weights):
        fig, ax = plt.subplots()
        ax.bar(labels, w.squeeze(), width)

        ax.set_xlabel('Samples')
        ax.set_ylabel('Weights')

        wandb.log({f"plot_weights_distribution_head_{i}": plt})


def plot_evals_distribution(evals):
    labels = list(range(evals[0].shape[0]))
    width = 0.9

    for i, w in enumerate(evals):
        fig, ax = plt.subplots()
        ax.bar(labels, w.squeeze(), width)

        ax.set_xlabel('Samples')
        ax.set_ylabel('Evals')

        wandb.log({f"plot_evals_distribution_head_{i}": plt})
