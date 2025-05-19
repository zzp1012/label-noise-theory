import os, torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple

# import internal libs
from utils import get_logger

def load(root: str = "../data",
         normalize: bool=True,
         randomize: bool=True) -> Tuple[Dataset, Dataset]:
    """load the cifar10 dataset.
    Args:
        root (str): the root path of the dataset.
        normalize (bool): whether to normalize the dataset.
        randomize (bool): whether to randomize the dataset.
    Returns:
        return the dataset.
    """
    logger = get_logger(__name__)
    logger.info("loading cifar10...")

    # prepare the transform
    transform_train = transforms.Compose([
        *([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if randomize else []), 
        transforms.ToTensor(),
        *([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] if normalize else []),
    ])
    logger.info(f"transform_train: {transform_train}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        *([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] if normalize else []),
    ])
    logger.info(f"transform_test: {transform_test}")

    # load the dataset
    os.makedirs(root, exist_ok=True)
    trainset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)

    # show basic info of dataset
    logger.info(f"trainset size: {len(trainset)}")
    logger.info(f"testset size: {len(testset)}")
    return trainset, testset