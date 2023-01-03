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

    data = list(zip(labels,  # change to images if needed
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


def wandb_summary(model, state):
    if state['args'].model_class.endswith('dsob'):
        # splitting_weights(model, state)
        evals_vs_confidence(model, state)


def splitting_weights(model, state):
    model = copy.deepcopy(model)

    _, _, test_data = DATASETS_NAME_MAP[state['args'].dataset]()
    test_loader = get_loader(test_data, 32, shuffle=True)
    _, weights, _ = model(next(iter(test_loader))[0])

    weights = [torch.squeeze(w.detach()) for w in weights]

    labels = list(range(32))
    width = 0.9
    fig, ax = plt.subplots()

    y = torch.zeros_like(weights[0], requires_grad=False)
    for i, w in enumerate(weights):
        ax.bar(labels, w, width, bottom=y, label=f'Head {i}')
        y += w

    ax.set_ylabel('weights')
    ax.set_title('Splitting weights')

    wandb.log({"split weights": plt}, commit=True)


def evals_vs_confidence(model, state):
    model = copy.deepcopy(model)

    _, _, test_data = DATASETS_NAME_MAP[state['args'].dataset]()
    test_loader = get_loader(test_data, 256, shuffle=True)
    head_preds, labels = get_preds_earlyexiting(model,
                                                test_loader,
                                                batches=0)

    _, _, evals = model(next(iter(test_loader))[0])
    head_preds = [nn.Softmax(dim=1)(x) for x in head_preds]
    confidence = [torch.max(x, dim=1).values for x in head_preds]
    output = [torch.max(x, dim=1).indices for x in head_preds]

    data = list(zip(labels,  # change to images if needed
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

    correct = [0] * 4
    for i in range(4):
        for d in data_head[i]:
            if d[3 + 3 * i] == d[13]:
                correct[i] += 1

    wandb.log({"eval vs conf: splitting conf head 0": correct[0] / 64,
               "eval vs conf: splitting conf head 1": correct[1] / 64,
               "eval vs conf: splitting conf head 2": correct[2] / 64,
               "eval vs conf: splitting conf head 3": correct[3] / 64,
               "eval vs conf: splitting conf avg": (correct[0] + correct[1] + correct[2] + correct[3]) / 256},
              commit=False)

    data = list(zip(labels,  # change to images if needed
                    head_preds[0],
                    evals[0],
                    output[0],
                    head_preds[1],
                    evals[1],
                    output[1],
                    head_preds[2],
                    evals[2],
                    output[2],
                    head_preds[3],
                    evals[3],
                    output[3],
                    labels))

    data_tmp = data
    data_head = [None] * 4
    for i in range(4):
        data_tmp.sort(key=lambda x: x[2 + 3 * i], reverse=True)
        data_head[i], data_tmp = data_tmp[:64], data_tmp[64:]

    correct = [0] * 4
    for i in range(4):
        for d in data_head[i]:
            if d[3 + 3 * i] == d[13]:
                correct[i] += 1

    wandb.log({"eval vs conf: splitting eval head 0": correct[0] / 64,
               "eval vs conf: splitting eval head 1": correct[1] / 64,
               "eval vs conf: splitting eval head 2": correct[2] / 64,
               "eval vs conf: splitting eval head 3": correct[3] / 64,
               "eval vs conf: splitting eval avg": (correct[0] + correct[1] + correct[2] + correct[3]) / 256},
              commit=False)

    def get_coef(e, c):
        e = torch.squeeze(e)
        e = torch.unsqueeze(e, 0)
        c = torch.unsqueeze(c, 0)
        return torch.corrcoef(torch.cat((e, c), 0))[0, 1]

    wandb.log({"eval vs conf: corrcoef head 0": get_coef(evals[0], confidence[0]),
               "eval vs conf: corrcoef head 1": get_coef(evals[1], confidence[1]),
               "eval vs conf: corrcoef head 2": get_coef(evals[2], confidence[2]),
               "eval vs conf: corrcoef head 3": get_coef(evals[3], confidence[3])},
              commit=True)
