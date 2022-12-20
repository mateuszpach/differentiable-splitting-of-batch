import copy

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn

from datasets import DATASETS_NAME_MAP
from eval import get_preds_earlyexiting
from utils import get_loader


def wandb_stats(test_loss, test_acc, train_loss, train_acc, model, state):
    accuracy_per_head(state)
    loss_per_head(state)
    splitting(model, state)

    wandb.log({"test_loss": sum(test_loss), "train_loss": sum(train_loss)}, commit=False)
    wandb.log({"test_acc": sum(test_acc) / len(test_acc), "train_acc": sum(train_acc) / len(train_acc)}, commit=True)
    # last MUST be 'commit=true'


def accuracy_per_head(state):
    colors = 'bgrc'
    for head_id in range(4):
        plt.plot([x[head_id] for x in state['test_acc']], colors[head_id] + '-')
        plt.plot([x[head_id] for x in state['train_acc']], colors[head_id] + '--')

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend([f'{y} Head {x}' for x in range(4) for y in ['Test', 'Train']], loc='upper left')
    wandb.log({"accuracy per head": plt}, commit=False)


def loss_per_head(state):
    colors = 'bgrc'
    for head_id in range(4):
        plt.plot([x[head_id].cpu() for x in state['test_loss']], colors[head_id] + '-')
        plt.plot([x[head_id].cpu() for x in state['train_loss']], colors[head_id] + '--')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend([f'{y} Head {x}' for x in range(4) for y in ['Test', 'Train']], loc='upper left')
    wandb.log({"loss per head": plt}, commit=False)


def splitting(model, state):
    model = copy.deepcopy(model)

    _, _, test_data = DATASETS_NAME_MAP[state['args'].dataset]()
    test_loader = get_loader(test_data, 256, shuffle=True)
    head_preds, labels = get_preds_earlyexiting(model,
                                                test_loader,
                                                batches=0)

    # _, _, test_data = DATASETS_NAME_MAP[state['args'].dataset](normalize=False)
    # test_loader = get_loader(test_data, 256, shuffle=True)
    # images, _ = next(iter(test_loader))

    head_preds = [nn.Softmax(dim=1)(x) for x in head_preds]
    confidence = [torch.max(x, dim=1).values for x in head_preds]
    output = [torch.max(x, dim=1).indices for x in head_preds]

    data = list(zip(labels, # change to images if needed
                    head_preds[0],
                    confidence[0],
                    output[0],
                    head_preds[1],
                    confidence[1],
                    output[1],
                    head_preds[2],
                    confidence[2],
                    output[2],
                    head_preds[3],
                    confidence[3],
                    output[3],
                    labels))

    data_tmp = data
    data_head = [None] * 4
    for i in range(4):
        data_tmp.sort(key=lambda x: x[2 + 3 * i], reverse=True)
        data_head[i], data_tmp = data_tmp[:64], data_tmp[64:]

    # fig = plt.figure(figsize=(8, 8))
    # for i in range(4):
    #     for j in range(1, 10):
    #         img = data_head[i][j][0]
    #         fig.add_subplot(4, 10, i * 10 + j)
    #         plt.imshow(img.permute(1, 2, 0))
    # fig.suptitle('Top 10 by confidence for each head on confidence-based split batch \n')
    # wandb.log({"Top 10": plt}, commit=False)

    correct = [0] * 4
    for i in range(4):
        for d in data_head[i]:
            if d[3 + 3 * i] == d[13]:
                correct[i] += 1

    wandb.log({"splitting conf head 0": correct[0] / 64,
               "splitting conf head 1": correct[1] / 64,
               "splitting conf head 2": correct[2] / 64,
               "splitting conf head 3": correct[3] / 64,
               "splitting conf avg": (correct[0] + correct[1] + correct[2] + correct[3]) / 256}, commit=False)

    correct = [0] * 4

    for d in data:
        for i in range(4):
            if d[3 + 3 * i] == d[13]:
                correct[i] += 1

    wandb.log({"splitting head 0": correct[0] / 256,
               "splitting head 1": correct[1] / 256,
               "splitting head 2": correct[2] / 256,
               "splitting head 3": correct[3] / 256,
               "splitting avg": (correct[0] + correct[1] + correct[2] + correct[3]) / 256 / 4}, commit=False)


def main():
    torch.set_printoptions(edgeitems=5, sci_mode=False, linewidth=200)

    # learning_stats(Path.cwd() / 'runs', 'cifar10_resnet18_frozen_4heads_EQMIVFDV_1')
    # learning_stats(Path.cwd() / 'runs', 'cifar10_resnet18_4heads_PIUVA74Z_2')
    # learning_stats(Path.cwd() / 'runs', 'mnist_resnet18_4heads_YBHRBB24_2')
    # learning_stats(Path.cwd() / 'runs', 'mnist_resnet18_frozen_4heads_HQ6YUQG6_1')
    # learning_stats(Path.cwd() / 'runs', 'cifar10_resnet18_4heads_PIUVA74Z_3')

    # heads_stats(Path.cwd() / 'runs', 'cifar10_resnet18_frozen_4heads_EQMIVFDV_1')
    # heads_stats(Path.cwd() / 'runs', 'cifar10_resnet18_4heads_PIUVA74Z_2')
    # heads_stats(Path.cwd() / 'runs', 'mnist_resnet18_4heads_YBHRBB24_2')
    # heads_stats(Path.cwd() / 'runs', 'mnist_resnet18_frozen_4heads_HQ6YUQG6_1')
    # heads_stats(Path.cwd() / 'runs', 'cifar10_resnet18_4heads_PIUVA74Z_3')


if __name__ == '__main__':
    main()
