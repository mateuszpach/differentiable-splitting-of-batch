import os

from torch.utils.data import Subset
from torchvision import datasets, transforms

DOWNLOAD = False


def get_mnist():
    dataset_path = os.environ['MNIST_PATH']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_data = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = Subset(train_data, list(range(10000)))
    test_data = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['CIFAR10_PATH']
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])

    train_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = Subset(datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval), list(range(10000)))
    test_data = datasets.CIFAR10(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


DATASETS_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10
}
