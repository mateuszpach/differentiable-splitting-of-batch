import json
import logging
import math
import os

import torch
import wandb

from common import parser, LOSS_NAME_MAP, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP
from datasets import DATASETS_NAME_MAP
from eval import test_earlyexiting_classification
from models import MODEL_NAME_MAP
from plots import wandb_stats
import utils


def train(args):
    model = MODEL_NAME_MAP[args.model_class]().to(utils.get_device())
    train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset]()
    batch_size = args.batch_size
    train_loader, train_eval_loader = utils.get_loader(train_data, batch_size), utils.get_loader(train_eval_data,
                                                                                                 batch_size)
    test_loader = utils.get_loader(test_data, batch_size)

    criterion_args = json.loads(args.loss_args)
    criterion_type = LOSS_NAME_MAP[args.loss_type]
    criterion = criterion_type(reduction='none', **criterion_args)
    optimizer_args = json.loads(args.optimizer_args)
    optimizer = OPTIMIZER_NAME_MAP[args.optimizer_class](model.parameters(), **optimizer_args)
    if args.scheduler_class is not None:
        scheduler_args = json.loads(args.scheduler_args)
        if 'T_0' in scheduler_args:
            scheduler_args['T_0'] = int(math.ceil(scheduler_args['T_0'] * (args.epochs - 1)))
        if 'patience' in scheduler_args:
            scheduler_args['patience'] = int(scheduler_args['patience'] * (args.epochs - 1))
        if args.scheduler_class == 'cosine':
            scheduler_args['T_max'] = args.epochs - 1
        scheduler = SCHEDULER_NAME_MAP[args.scheduler_class](optimizer, **scheduler_args)
    else:
        scheduler = None

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    _, run_name = utils.generate_run_name(args)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f'state.pth'

    entity = os.environ['WANDB_ENTITY']
    project = os.environ['WANDB_PROJECT']
    run_id = utils.get_run_id(run_name)
    if run_id is not None:
        wandb.init(entity=entity, project=project, id=run_id, resume='must', dir=str(run_dir.resolve()))
    else:
        wandb.init(entity=entity, project=project, config=args, name=run_name, dir=str(run_dir.resolve()))
    wandb.run.log_code('.', include_fn=lambda path: path.endswith('.py'))

    try:
        state = utils.load_state(state_path)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler_state'])
        trained_epochs = state['trained_epochs']
    except FileNotFoundError:
        state = {'args': args}
        trained_epochs = 0
        state['test_loss'] = []
        state['test_acc'] = []
        state['train_loss'] = []
        state['train_acc'] = []

    model.train()

    for epoch in range(trained_epochs, args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(utils.get_device(), non_blocking=True)
            labels = labels.to(utils.get_device(), non_blocking=True)

            outputs, consume_weights = model(images)

            loss = [criterion(head_outputs, labels) for head_outputs in outputs]

            if consume_weights:
                loss = [l * w for l, w in zip(loss, consume_weights)]

            loss = sum([torch.mean(l) for l in loss])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss, test_acc = test_earlyexiting_classification(model,
                                                               test_loader,
                                                               criterion_type,
                                                               batches=args.eval_batches)
        train_loss, train_acc = test_earlyexiting_classification(model,
                                                                 train_eval_loader,
                                                                 criterion_type,
                                                                 batches=args.eval_batches)

        if scheduler is not None:
            if args.scheduler_class == 'reduce_on_plateau':
                scheduler.step(sum(train_loss))
            else:
                scheduler.step()

        state['trained_epochs'] = epoch + 1
        state['model_state'] = model.state_dict()
        state['optimizer_state'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state'] = scheduler.state_dict()
        state['test_loss'].append([x.cpu() for x in test_loss])
        state['test_acc'].append(test_acc)
        state['train_loss'].append([x.cpu() for x in train_loss])
        state['train_acc'].append(train_acc)

        wandb_stats(test_loss, test_acc, train_loss, train_acc, model, state)

        utils.save_state(state, state_path)
        utils.save_state(state, state_path.parent / f'{state_path.stem}_epoch{epoch + 1}.pth')

        # logging.info(f'Epoch {epoch + 1}/{args.epochs} finished with\n'
        #              f'test_loss={test_loss}\n'
        #              f'test_acc={[f"{x:.3f}" for x in test_acc]}\n'
        #              f'train_loss={train_loss}\n'
        #              f'train_acc={[f"{x:.3f}" for x in train_acc]}')


def main():
    args = parser.parse_args()
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging.')
    train(args)


if __name__ == '__main__':
    main()
