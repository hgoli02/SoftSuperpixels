from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import Caltech101
import time
import scipy.ndimage as nd
import numpy as np
import imageio
import skimage.io as io

# Point this to your *foldered* val set, e.g. /mnt/ssd/users/hossein/val_cls
IMAGENET_VAL_DIR = "/mnt/ssd/users/hossein/val_cls"

# list of all datasets
DATASETS = ["imagenet", "cifar10", "imagenet_10", "caltech101", "intel", "tiny", "mnist"]

# You have to populate the ImageNet and Intel datasets. Other datasets will be downloaded if not available.

def get_dataset(dataset: str, split: str, device='cuda:0') -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "imagenet_10":
        return _imagenet10(split)
    elif dataset == 'caltech101':
        return _caltech(split)
    elif dataset == 'intel':
        return _intel(split)
    elif dataset == 'tiny':
        return _tiny(split)
    elif dataset == 'mnist':
        return datasets.MNIST("./dataset_cache", train=split == "train", download=True, transform=transforms.ToTensor())

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset."""
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "imagenet_10":
        return 10
    elif dataset == "caltech101":
        return 101
    elif dataset == 'intel':
        return 6
    elif dataset == 'tiny':
        return 200
    elif dataset == 'mnist':
        return 10

def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "imagenet_10":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == 'caltech101':
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == 'intel':
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == 'tiny':
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == 'mnist':
        return NormalizeLayer([0.1307], [0.3081])

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.0, 0.0, 0.0]
_CIFAR10_STDDEV = [1, 1, 1]

def _tiny(split: str) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if split == "train":
        return datasets.ImageFolder("./tiny/train", transform=transform)
    elif split == "test":
        return datasets.ImageFolder("./tiny/test", transform=transform)

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _imagenet(split: str) -> Dataset:
    """
    ImageNet-1k *validation* loader.
    Assumes IMAGENET_VAL_DIR has class subfolders (WNIDs), e.g.: n01440764/*.JPEG
    """
    if split == "train":
        raise RuntimeError("Only the validation set is supported here (no train images provided).")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_root = IMAGENET_VAL_DIR
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"IMAGENET_VAL_DIR not found or not a directory: {val_root}")
    # Directly use ImageFolder; no manual relabeling needed.
    return datasets.ImageFolder(val_root, transform=transform)

def _imagenet10(split: str) -> Dataset:
    transform_old = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if split == "train":
        return datasets.ImageFolder("./imagenette2/train", transform_old)
    elif split == "test":
        return datasets.ImageFolder("./imagenette2/val", transform_old)

def _caltech(split: str) -> Dataset:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if split == "train":
        return Caltech101("./dataset_cache", train=True, download=True, transform=transform)
    elif split == "test":
        return Caltech101("./dataset_cache", train=False, download=True, transform=transform)

def _intel(split: str) -> Dataset:
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    if split == "train":
        return datasets.ImageFolder("./intel/seg_train/seg_train", transform=transform)
    elif split == "test":
        return datasets.ImageFolder("./intel/seg_test/seg_test", transform=transform)

class NormalizeLayer(torch.nn.Module):
    """
    Standardize channels by subtracting dataset mean and dividing by std.
    Registered as buffers so it auto-moves with .to(device).
    """
    def __init__(self, means: List[float], sds: List[float]):
        super().__init__()
        m = torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1)
        s = torch.tensor(sds, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("means", m)
        self.register_buffer("sds", s)

    def forward(self, x: torch.Tensor):
        return (x - self.means) / self.sds
