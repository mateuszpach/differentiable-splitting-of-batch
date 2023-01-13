from random import shuffle
from typing import List, Tuple

import torch
import wandb

from plots import plot_weights_distribution_stacked, plot_weights_distribution
from utils import get_device


def run_model(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              device: torch.device = get_device()) -> Tuple[List[torch.Tensor], torch.Tensor]:
    saved_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch, (images, labels) in enumerate(data_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs, consume_weights, evals = model(images)
                outputs = [head_outputs.detach().cpu() for head_outputs in outputs]
                consume_weights = [head_consume_weights.detach().cpu() for head_consume_weights in consume_weights]
                evals = [head_evals.detach().cpu() for head_evals in evals]
                yield outputs, consume_weights, evals, labels.cpu()
    finally:
        model.train(saved_training)


def evaluate(model, criterion, data_loader, dataset_name, epoch, gammas, ext_thresholds_conf=None,
             ext_thresholds_eval=None):
    batches = 0

    losses = [0] * model.number_of_heads
    losses_sum = 0
    weighted_losses = [0] * model.number_of_heads
    weighted_losses_sum = 0
    accuracies = [0] * model.number_of_heads
    accuracies_mean = 0
    weighted_accuracies = [0] * model.number_of_heads
    weighted_accuracies_mean = 0
    accuracies_conf = [0] * model.number_of_heads
    accuracies_conf_mean = 0
    thresholds_conf = [0] * model.number_of_heads
    accuracies_eval = [0] * model.number_of_heads
    accuracies_eval_mean = 0
    thresholds_eval = [0] * model.number_of_heads
    accuracies_rand = [0] * model.number_of_heads
    accuracies_rand_mean = 0
    accuracies_conf_thresh = [0] * model.number_of_heads
    accuracies_conf_thresh_mean = 0
    sizes_conf_thresh = [0] * model.number_of_heads
    sizes_conf_thresh_mean = 0
    accuracies_eval_thresh = [0] * model.number_of_heads
    accuracies_eval_thresh_mean = 0
    sizes_eval_thresh = [0] * model.number_of_heads
    sizes_eval_thresh_mean = 0
    eval_conf_corr = [0] * model.number_of_heads
    eval_conf_corr_mean = 0

    for batch_i, (outputs, consume_weights, evals, labels) in enumerate(run_model(model, data_loader)):
        batches += 1
        batch_size = labels.size(0)

        # note: we round instead ceiling, so it is important that last gamma is None when dealing with floats
        chunk_sizes = [round(batch_size * gamma) for gamma in gammas if gamma is not None]
        if gammas[-1] is None:
            chunk_sizes.append(batch_size - sum(chunk_sizes))

        # losses
        b_losses_na = [criterion(head_outputs, labels) for head_outputs in outputs]
        b_losses = [torch.mean(l).item() for l in b_losses_na]
        b_losses_sum = sum(b_losses)

        b_weighted_losses_na = [l * w for l, w in zip(b_losses, consume_weights)]
        b_weighted_losses = [torch.mean(l).item() for l in b_weighted_losses_na]
        b_weighted_losses_sum = sum(b_weighted_losses)

        # accuracies
        b_preds = [head_outputs.argmax(dim=1) for head_outputs in outputs]
        b_accuracies = [(head_preds == labels).sum().item() / batch_size for head_preds in b_preds]
        b_accuracies_mean = sum(b_accuracies) / len(b_accuracies)

        b_weighted_accuracies = [((head_preds == labels).long() * w).sum().item() / (batch_size * w.sum()) for
                                 head_preds, w in zip(b_preds, consume_weights)]
        b_weighted_accuracies_mean = sum(b_weighted_accuracies) / len(b_weighted_accuracies)

        # splitting by confidence: batch is split into chunks (with gamma proportions),
        # we iterate over heads, head_i takes top gamma_i% samples by confidence and leaves rest for other heads
        # we measure accuracy of the heads on respective chunks
        confidences = [head_outputs.max(dim=1).values for head_outputs in outputs]

        b_accuracies_conf = []
        b_thresholds_conf = []
        indices = list(range(batch_size))
        for chunk_i, chunk_size in enumerate(chunk_sizes):
            indices.sort(key=lambda x: confidences[chunk_i][x], reverse=True)
            chunk_preds = torch.stack([b_preds[chunk_i][i] for i in indices[:chunk_size]])
            chunk_labels = torch.stack([labels[i] for i in indices[:chunk_size]])
            b_accuracies_conf.append((chunk_preds == chunk_labels).sum().item() / chunk_size)
            b_thresholds_conf.append(confidences[chunk_i][indices[chunk_size - 1]])
            indices = indices[chunk_size:]
        b_accuracies_conf_mean = sum(b_accuracies_conf) / len(b_accuracies_conf)

        # splitting by eval: analogous to splitting by confidence, but heads take by eval
        b_accuracies_eval = []
        b_thresholds_eval = []
        indices = list(range(batch_size))
        for chunk_i, chunk_size in enumerate(chunk_sizes):
            indices.sort(key=lambda x: evals[chunk_i][x], reverse=True)
            chunk_preds = torch.stack([b_preds[chunk_i][i] for i in indices[:chunk_size]])
            chunk_labels = torch.stack([labels[i] for i in indices[:chunk_size]])
            b_accuracies_eval.append((chunk_preds == chunk_labels).sum().item() / chunk_size)
            b_thresholds_eval.append(evals[chunk_i][indices[chunk_size - 1]])
            indices = indices[chunk_size:]
        b_accuracies_eval_mean = sum(b_accuracies_eval) / len(b_accuracies_eval)

        # splitting randomly: analogous to splitting by confidence, but heads take randomly
        b_accuracies_rand = []
        indices = list(range(batch_size))
        shuffle(indices)
        for chunk_i, chunk_size in enumerate(chunk_sizes):
            chunk_preds = torch.stack([b_preds[chunk_i][i] for i in indices[:chunk_size]])
            chunk_labels = torch.stack([labels[i] for i in indices[:chunk_size]])
            b_accuracies_rand.append((chunk_preds == chunk_labels).sum().item() / chunk_size)
            indices = indices[chunk_size:]
        b_accuracies_rand_mean = sum(b_accuracies_rand) / len(b_accuracies_rand)

        # splitting by confidence thresholds: batch is split into chunks, we iterate over heads,
        # head_i takes samples with confidence > ext_thresholds_conf[i] and leaves rest for other heads
        # we measure accuracy of the heads on respective chunks
        b_accuracies_conf_thresh = []
        b_sizes_conf_thresh = []
        indices = list(range(batch_size))
        for chunk_i, threshold in enumerate(ext_thresholds_conf if ext_thresholds_conf else b_thresholds_conf):
            indices.sort(key=lambda x: confidences[chunk_i][x], reverse=True)
            chunk_size = len([i for i in indices if confidences[chunk_i][i] >= threshold])
            chunk_preds = torch.stack([b_preds[chunk_i][i] for i in indices[:chunk_size]])
            chunk_labels = torch.stack([labels[i] for i in indices[:chunk_size]])
            b_accuracies_conf_thresh.append((chunk_preds == chunk_labels).sum().item() / chunk_size)
            b_sizes_conf_thresh.append(chunk_size)
            indices = indices[chunk_size:]
        b_accuracies_conf_thresh_mean = sum(b_accuracies_conf_thresh) / len(b_accuracies_conf_thresh)
        b_sizes_conf_thresh_mean = sum(b_sizes_conf_thresh) / len(b_sizes_conf_thresh)

        # splitting by eval thresholds: analogous to splitting by confidence thresholds, but eval
        b_accuracies_eval_thresh = []
        b_sizes_eval_thresh = []
        indices = list(range(batch_size))
        for chunk_i, threshold in enumerate(ext_thresholds_eval if ext_thresholds_eval else b_thresholds_eval):
            indices.sort(key=lambda x: evals[chunk_i][x], reverse=True)
            chunk_size = len([i for i in indices if evals[chunk_i][i] >= threshold])
            chunk_preds = torch.stack([b_preds[chunk_i][i] for i in indices[:chunk_size]])
            chunk_labels = torch.stack([labels[i] for i in indices[:chunk_size]])
            b_accuracies_eval_thresh.append((chunk_preds == chunk_labels).sum().item() / chunk_size)
            b_sizes_eval_thresh.append(chunk_size)
            indices = indices[chunk_size:]
        b_accuracies_eval_thresh_mean = sum(b_accuracies_eval_thresh) / len(b_accuracies_eval_thresh)
        b_sizes_eval_thresh_mean = sum(b_sizes_eval_thresh) / len(b_sizes_eval_thresh)

        # correlations between eval and confidence
        b_eval_conf_corr = []
        for head_i, (head_evals, head_confidences) in enumerate(zip(evals, confidences)):
            e = torch.unsqueeze(torch.squeeze(head_evals), 0)
            c = torch.unsqueeze(head_confidences, 0)
            b_eval_conf_corr.append(torch.corrcoef(torch.cat((e, c), 0))[0, 1])
        b_eval_conf_corr_mean = sum(b_eval_conf_corr) / len(b_eval_conf_corr)

        # sum all metrics over batches
        losses = [sum(x) for x in zip(losses, b_losses)]
        losses_sum += b_losses_sum

        weighted_losses = [sum(x) for x in zip(weighted_losses, b_weighted_losses)]
        weighted_losses_sum += b_weighted_losses_sum

        accuracies = [sum(x) for x in zip(accuracies, b_accuracies)]
        accuracies_mean += b_accuracies_mean

        weighted_accuracies = [sum(x) for x in zip(weighted_accuracies, b_weighted_accuracies)]
        weighted_accuracies_mean += b_weighted_accuracies_mean

        accuracies_conf = [sum(x) for x in zip(accuracies_conf, b_accuracies_conf)]
        accuracies_conf_mean += b_accuracies_conf_mean
        thresholds_conf = [sum(x) for x in zip(thresholds_conf, b_thresholds_conf)]

        accuracies_eval = [sum(x) for x in zip(accuracies_eval, b_accuracies_eval)]
        accuracies_eval_mean += b_accuracies_eval_mean
        thresholds_eval = [sum(x) for x in zip(thresholds_eval, b_thresholds_eval)]

        accuracies_rand = [sum(x) for x in zip(accuracies_rand, b_accuracies_rand)]
        accuracies_rand_mean += b_accuracies_rand_mean

        accuracies_conf_thresh = [sum(x) for x in zip(accuracies_conf_thresh, b_accuracies_conf_thresh)]
        accuracies_conf_thresh_mean += b_accuracies_conf_thresh_mean
        sizes_conf_thresh = [sum(x) for x in zip(sizes_conf_thresh, b_sizes_conf_thresh)]
        sizes_conf_thresh_mean += b_sizes_conf_thresh_mean

        accuracies_eval_thresh = [sum(x) for x in zip(accuracies_eval_thresh, b_accuracies_eval_thresh)]
        accuracies_eval_thresh_mean += b_accuracies_eval_thresh_mean
        sizes_eval_thresh = [sum(x) for x in zip(sizes_eval_thresh, b_sizes_eval_thresh)]
        sizes_eval_thresh_mean += b_sizes_eval_thresh_mean

        eval_conf_corr = [sum(x) for x in zip(eval_conf_corr, b_eval_conf_corr)]
        eval_conf_corr_mean += b_eval_conf_corr_mean

        # single batch metrics
        if batch_i == 0:
            consume_weights_shrank = [x[:30] for x in consume_weights]
            plot_weights_distribution_stacked(consume_weights_shrank, epoch)
            plot_weights_distribution(consume_weights_shrank, epoch)

    # divide by number of batches to get means over batches
    losses = [x / batches for x in losses]
    losses_sum /= batches

    weighted_losses = [x / batches for x in weighted_losses]
    weighted_accuracies_mean /= batches

    accuracies = [x / batches for x in accuracies]
    accuracies_mean /= batches

    weighted_accuracies = [x / batches for x in weighted_accuracies]
    weighted_accuracies_mean /= batches

    accuracies_conf = [x / batches for x in accuracies_conf]
    accuracies_conf_mean /= batches
    thresholds_conf = [x / batches for x in thresholds_conf]

    accuracies_eval = [x / batches for x in accuracies_eval]
    accuracies_eval_mean /= batches
    thresholds_eval = [x / batches for x in thresholds_eval]

    accuracies_rand = [x / batches for x in accuracies_rand]
    accuracies_rand_mean /= batches

    accuracies_conf_thresh = [x / batches for x in accuracies_conf_thresh]
    accuracies_conf_thresh_mean /= batches
    sizes_conf_thresh = [x / batches for x in sizes_conf_thresh]
    sizes_conf_thresh_mean /= batches

    accuracies_eval_thresh = [x / batches for x in accuracies_eval_thresh]
    accuracies_eval_thresh_mean /= batches
    sizes_eval_thresh = [x / batches for x in sizes_eval_thresh]
    sizes_eval_thresh_mean /= batches

    eval_conf_corr = [x / batches for x in eval_conf_corr]
    eval_conf_corr_mean /= batches

    # send to wandb
    for head_id in range(model.number_of_heads):
        wandb.log({f'{dataset_name}_loss_{head_id}': losses[head_id],
                   f'{dataset_name}_weighted_loss_{head_id}': weighted_losses[head_id],
                   f'{dataset_name}_accuracy_{head_id}': accuracies[head_id],
                   f'{dataset_name}_weighted_accuracy_{head_id}': weighted_accuracies[head_id],
                   f'{dataset_name}_accuracy_conf_{head_id}': accuracies_conf[head_id],
                   f'{dataset_name}_threshold_conf_{head_id}': thresholds_conf[head_id],
                   f'{dataset_name}_accuracy_eval_{head_id}': accuracies_eval[head_id],
                   f'{dataset_name}_threshold_eval_{head_id}': thresholds_eval[head_id],
                   f'{dataset_name}_accuracy_rand_{head_id}': accuracies_rand[head_id],
                   f'{dataset_name}_accuracy_conf_thresh_{head_id}': accuracies_conf_thresh[head_id],
                   f'{dataset_name}_size_conf_thresh_{head_id}': sizes_conf_thresh[head_id],
                   f'{dataset_name}_accuracy_eval_thresh_{head_id}': accuracies_eval_thresh[head_id],
                   f'{dataset_name}_size_eval_thresh_{head_id}': sizes_eval_thresh[head_id],
                   f'{dataset_name}_eval_conf_corr_{head_id}': eval_conf_corr[head_id]}, step=epoch)

    wandb.log({f'{dataset_name}_losses_sum': losses_sum,
               f'{dataset_name}_weighted_losses_sum': weighted_losses_sum,
               f'{dataset_name}_accuracies_mean': accuracies_mean,
               f'{dataset_name}_weighted_accuracies_mean': weighted_accuracies_mean,
               f'{dataset_name}_accuracies_conf_mean': accuracies_conf_mean,
               f'{dataset_name}_accuracies_eval_mean': accuracies_eval_mean,
               f'{dataset_name}_accuracies_rand_mean': accuracies_rand_mean,
               f'{dataset_name}_accuracies_conf_thresh_mean': accuracies_conf_thresh_mean,
               f'{dataset_name}_accuracies_eval_thresh_mean': accuracies_eval_thresh_mean,
               f'{dataset_name}_eval_conf_corr_mean': eval_conf_corr_mean}, step=epoch)

    return thresholds_conf, thresholds_eval


def evaluate_on_test_and_eval_datasets(model, criterion, test_loader, train_eval_loader, epoch,
                                       gammas=(0.25, 0.25, 0.25, None)):
    # evaluate on train dataset and get thresholds
    thresholds_conf, thresholds_eval = evaluate(model, criterion, train_eval_loader, 'train', epoch, gammas)
    # make sure the last head consumes the remainders
    thresholds_conf[-1] = 0
    thresholds_eval[-1] = 0
    # evaluate on test dataset using pre-calculated thresholds
    evaluate(model, criterion, test_loader, 'test', epoch, gammas, thresholds_conf, thresholds_eval)
    # evaluate(model, criterion, test_loader, 'test', epoch, gammas)
