from typing import List, Tuple

import torch

from utils import get_device


def get_preds_earlyexiting(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           batches: int = 0,
                           device: torch.device = get_device()) -> Tuple[List[torch.Tensor], torch.Tensor]:
    saved_training = model.training
    model.eval()
    batch_outputs = []
    batch_labels = []
    with torch.no_grad():
        # model.all_mode() ???
        for batch, (X, y) in enumerate(data_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            output, _ = model(X)
            y_preds = [y_pred.detach() for y_pred in output]
            batch_outputs.append(y_preds)
            batch_labels.append(y)
            if batch >= batches:
                break
    model.train(saved_training)
    head_preds = []
    for i in range(model.number_of_heads):
        head_outputs = torch.cat([batch_output[i] for batch_output in batch_outputs])
        head_preds.append(head_outputs)
    labels = torch.cat(batch_labels)
    return head_preds, labels


def test_earlyexiting_classification(model: torch.nn.Module,
                                     data_loader: torch.utils.data.DataLoader,
                                     criterion_class: torch.nn.Module,
                                     batches: int = 0,
                                     device: torch.device = get_device()) -> (float, float):
    criterion = criterion_class(reduction='mean')
    head_preds, ys = get_preds_earlyexiting(model, data_loader, batches, device)
    head_losses = [criterion(preds, ys) for preds in head_preds]
    head_accuracies = [(preds.argmax(dim=1) == ys).sum().item() / ys.size(0) for preds in head_preds]
    return head_losses, head_accuracies
