import os

import torch
from torchvision import datasets, transforms

DOWNLOAD = False


def get_mnist(normalize=True):
    dataset_path = os.environ['MNIST_PATH']
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    train_data = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = train_data
    test_data = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(proper_normalization=True, normalize=True):
    if proper_normalization:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['CIFAR10_PATH']
    if normalize:
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
    else:
        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1),
        ])

    train_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR10(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_mnist_cifar10(normalize=True):
    mnist_train, mnist_train_eval, mnist_test = get_mnist(normalize)
    cifar10_train, cifar10_train_eval, cifar10_test = get_cifar10(normalize)
    return torch.utils.data.ConcatDataset([mnist_train, cifar10_train]), \
           torch.utils.data.ConcatDataset([mnist_train_eval, cifar10_train_eval]), \
           torch.utils.data.ConcatDataset([mnist_test, cifar10_test])


def get_cifar100(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['CIFAR100_PATH']
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
    train_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR100(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_tinyimagenet(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['TINYIMAGENET_PATH']
    train_path = f'{dataset_path}/train'
    test_path = f'{dataset_path}/val/images'
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.ImageFolder(train_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(train_path, transform=transform_eval)
    test_data = datasets.ImageFolder(test_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


DATASETS_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
    'mnist_cifar10': get_mnist_cifar10
}
