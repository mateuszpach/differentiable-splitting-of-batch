import json
import logging
import math

import torch
import wandb

import utils
from common import parser, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP, LOSS_NAME_MAP
from datasets import DATASETS_NAME_MAP
from models import MODEL_NAME_MAP
from eval import evaluate_on_test_and_eval_datasets


def train(args):
    # load base model
    base_model = MODEL_NAME_MAP[args.model_class]().to(utils.get_device())
    state = utils.load_state(args.base_model_state_path)
    base_model.load_state_dict(state['model_state'])

    # load dataset
    train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset]()
    batch_size = args.batch_size
    train_loader = utils.get_loader(train_data, batch_size)
    train_eval_loader = utils.get_loader(train_eval_data, batch_size)
    test_loader = utils.get_loader(test_data, batch_size, shuffle=False)

    # setup path
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f'eet_{args.model_class}_{args.exp_id}'
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # init wandb
    utils.init_wandb(run_name, run_dir, args)

    # load model
    model = MODEL_NAME_MAP['expected_exit_time'](base_model).to(utils.get_device())

    # setup optimizer, scheduler
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

    # set device
    device = utils.get_device()

    # train loop
    model.train()
    for epoch in range(0, args.epochs):
        logging.info(f'Epoch {epoch} started')
        epoch_loss = 0

        for _, (imagesx, labelsx) in enumerate(train_loader):
            imagesx = imagesx.to(device, non_blocking=True)
            labelsx = labelsx.to(device, non_blocking=True)
            break
        # epoch loop
        for i, (images, labels) in enumerate(train_loader):
            # images = images.to(device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            images = imagesx
            labels = labelsx

            # all_logits: nr_of_heads x (batch_size, nr_classes)
            # all_consume_weights: nr_of_heads x (batch_size, 1)
            # all_evals: nr_of_heads x (batch_size, 1)
            # all_gammas: (nr_of_heads,)
            all_logits, all_consume_weights, all_evals, all_gammas = model(images)

            # base loss equal mean of consume_weights for incorrect preds and zeroes for correct preds
            all_preds = [torch.argmax(logits, dim=1) for logits in all_logits]
            base_losses = [preds == labels for preds in all_preds]
            # base_losses = [criterion(logits, labels) for logits in all_logits]
            weighted_base_losses = [l * w for l, w in zip(base_losses, all_consume_weights)]
            base_loss = 1 - torch.mean(torch.cat(weighted_base_losses))
            # base_loss = torch.mean(torch.cat(weighted_base_losses))

            # expected exit time loss
            current_expected_exit_time = torch.sum(all_gammas * torch.arange(1, model.number_of_heads + 1).to(device))
            eet_loss = args.eet_loss_factor * torch.square(current_expected_exit_time - args.expected_exit_time)
            # eet_loss = args.eet_loss_factor * torch.abs(current_expected_exit_time - args.expected_exit_time)

            loss = eet_loss + base_loss
            # if epoch == 0:
            #     loss = base_loss
            # else:
            #     epoch_progress = i / len(train_loader)
            #     loss = base_loss * (1 - epoch_progress) + eet_loss * epoch_progress
            # if eet_loss > 0.2:
            #     loss = eet_loss
            # else:
            #     loss = base_loss
            # if epoch < 1:
            #     loss = base_loss
            # else:
            #     loss = eet_loss

            # TODO: move it to eval.py
            wandb_step = int(100 * (epoch + (i / len(train_loader))))
            wandb.log({f'train_eet_loss': eet_loss,
                       f'train_base_loss': base_loss,
                       f'train_loss': loss}, step=wandb_step)

            # get metrics and send them to wandb
            if i == len(train_loader) // 2 or i == len(train_loader) - 1:
                logging.info(f'Epoch {epoch} evaluation started')
                all_gammas = [gamma.item() for gamma in all_gammas]
                # consume remainders
                all_gammas[-1] = None
                evaluate_on_test_and_eval_datasets(model, None, test_loader, train_eval_loader, wandb_step, all_gammas)
                logging.info(f'Epoch {epoch} evaluation finished')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        # end of epoch loop

        # update scheduler
        if scheduler is not None:
            if args.scheduler_class == 'reduce_on_plateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        logging.info(f'Epoch {epoch} finished')
    # end of train loop
    wandb.log({f'xd': 1}, step=1000000)

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
