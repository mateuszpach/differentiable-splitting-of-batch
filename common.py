import argparse
from pathlib import Path

from torch import optim, nn


def get_lrs(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


parser = argparse.ArgumentParser()
parser.add_argument('--exp_id',
                    help='experiment id',
                    type=int,
                    default=0)
parser.add_argument('--runs_dir',
                    help='directory to save the results to',
                    type=Path,
                    default=Path.cwd() / 'runs')
parser.add_argument('--model_class',
                    help='class of the model to train',
                    type=str,
                    required=True)
parser.add_argument('--dataset',
                    help='dataset to train on',
                    type=str,
                    required=True)
parser.add_argument('--epochs',
                    help='number of epochs to train for',
                    type=int,
                    required=True)
parser.add_argument('--batch_size',
                    help='batch size for training',
                    type=int)
parser.add_argument('--loss_type',
                    help='loss function to be used for training',
                    type=str)
parser.add_argument('--loss_args',
                    help='arguments to be passed to the loss init function',
                    type=str)
parser.add_argument('--optimizer_class',
                    help='class of the optimizer to use for training',
                    type=str)
parser.add_argument('--optimizer_args',
                    help='arguments to be passed to the optimizer init function',
                    type=str)
parser.add_argument('--scheduler_class',
                    help='class of the scheduler to use for training',
                    type=str,
                    default=None)
parser.add_argument('--scheduler_args',
                    help='arguments to be passed to the scheduler init function',
                    type=str)
parser.add_argument('--epochs_per_eval',
                    help='number of epochs between model evaluations',
                    type=int,
                    default=1)

LOSS_NAME_MAP = {
    'ce': nn.CrossEntropyLoss,
    'nll': nn.NLLLoss,
}

OPTIMIZER_NAME_MAP = {
    'sgd': optim.SGD,
    'adam': optim.AdamW,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'step_lr':  optim.lr_scheduler.StepLR
}
