import json
import logging
import math

import torch

import utils
from common import parser, LOSS_NAME_MAP, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP
from datasets import DATASETS_NAME_MAP
from eval import evaluate_on_test_and_eval_datasets
from models import MODEL_NAME_MAP


def train(args):
    # load model class and dataset
    model = MODEL_NAME_MAP[args.model_class]().to(utils.get_device())
    train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset]()
    batch_size = args.batch_size
    train_loader = utils.get_loader(train_data, batch_size)
    train_eval_loader = utils.get_loader(train_eval_data, batch_size)
    test_loader = utils.get_loader(test_data, batch_size, shuffle=False)

    # setup criterion, optimizer, scheduler
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

    # setup path
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    _, run_name = utils.generate_run_name(args)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f'state.pth'

    # init wandb
    utils.init_wandb(run_name, run_dir, args)

    # try loading saved state
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

    # train loop
    model.train()
    for epoch in range(trained_epochs, args.epochs):
        logging.info(f'Epoch {epoch} started')
        epoch_loss = 0

        # epoch loop
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(utils.get_device(), non_blocking=True)
            labels = labels.to(utils.get_device(), non_blocking=True)

            # outputs: list of nr_heads tensors, each tensor of size (batch_size, nr_classes)
            #          probabilities of matching given class in each head
            # consume_weights: list of nr_heads tensors, each tensor of size (batch_size, 1)
            #                  how much of each sample should be consumed in each head
            # if epoch >= 100:
            #     outputs, consume_weights, _ = model(images, min((epoch - 50) / (args.epochs - 51) + 0.1, 1))
            # else:
            #     outputs, consume_weights, _ = model(images, 1)
            outputs, consume_weights, _ = model(images)

            # for each head and sample calculate criterion loss and weigh it by consume_weights
            losses = [criterion(head_outputs, labels) for head_outputs in outputs]
            if consume_weights:
                # sf = (epoch - 50) / (args.epochs - 51)
                # losses = [l * w * sf + l * (1 - sf) for l, w in zip(losses, consume_weights)]
                losses = [l * w for l, w in zip(losses, consume_weights)]
            # for each head aggregate the scaled criterion losses by taking mean, and sum all heads losses
            loss = sum([torch.mean(l) for l in losses]) / len(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        # end of epoch loop

        # get metrics and send them to wandb
        if epoch % args.epochs_per_eval == 0:
            logging.info(f'Epoch {epoch} evaluation started')
            evaluate_on_test_and_eval_datasets(model, criterion, test_loader, train_eval_loader, epoch)
            logging.info(f'Epoch {epoch} evaluation finished')

        # update scheduler
        if scheduler is not None:
            if args.scheduler_class == 'reduce_on_plateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        # save state
        state['trained_epochs'] = epoch + 1
        state['model_state'] = model.state_dict()
        state['optimizer_state'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state'] = scheduler.state_dict()
        utils.save_state(state, state_path)

        # save intermediate step
        # utils.save_state(state, state_path.parent / f'{state_path.stem}_epoch{epoch + 1}.pth')

        logging.info(f'Epoch {epoch} finished')

    # end of train loop


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
