#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset loading. Everything is kept in memory as tensors (fast on CPU).

MNIST is loaded with a tiny built-in IDX reader and several download mirrors, so only
`torch` + `numpy` are needed. CIFAR-10 (optional) uses torchvision.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

MNIST_FILES = {
    'train_x': 'train-images-idx3-ubyte.gz',
    'train_y': 'train-labels-idx1-ubyte.gz',
    'test_x': 't10k-images-idx3-ubyte.gz',
    'test_y': 't10k-labels-idx1-ubyte.gz',
}
MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://raw.githubusercontent.com/fgnt/mnist/master/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
    'https://yann.lecun.com/exdb/mnist/',
]


class TensorData(Dataset):
    """In-memory dataset: x (N,C,H,W) float32, y (N,) int64."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.targets = y.numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _download(fname, raw_dir):
    path = os.path.join(raw_dir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(raw_dir, exist_ok=True)
    errors = []
    for mirror in MNIST_MIRRORS:
        url = mirror + fname
        try:
            print('Downloading', url)
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            if len(blob) < 1000:
                raise IOError('file too small')
            with open(path, 'wb') as f:
                f.write(blob)
            return path
        except Exception as e:  # try next mirror
            errors.append('%s -> %s' % (url, e))
    raise RuntimeError('Could not download %s from any mirror:\n  %s\n'
                       'Download the 4 MNIST .gz files manually into %s' % (fname, '\n  '.join(errors), raw_dir))


def _read_idx(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    magic = struct.unpack('>I', data[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack('>' + 'I' * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(data_dir):
    raw_dir = os.path.join(data_dir, 'mnist', 'raw')
    arrs = {k: _read_idx(_download(v, raw_dir)) for k, v in MNIST_FILES.items()}

    def prep(x, y):
        x = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
        x = (x - 0.1307) / 0.3081
        return TensorData(x, torch.from_numpy(y.astype(np.int64)))

    return prep(arrs['train_x'], arrs['train_y']), prep(arrs['test_x'], arrs['test_y'])


def load_cifar(data_dir):
    try:
        from torchvision import datasets
    except ImportError:
        raise SystemExit("CIFAR-10 needs torchvision: pip install torchvision")
    root = os.path.join(data_dir, 'cifar')
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)

    def prep(ds):
        x = torch.from_numpy(ds.data.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        x = (x - 0.5) / 0.5
        return TensorData(x.contiguous(), torch.tensor(ds.targets, dtype=torch.int64))

    return prep(tr), prep(te)


def load_dataset(name, data_dir):
    if name == 'mnist':
        return load_mnist(data_dir)
    if name == 'cifar':
        return load_cifar(data_dir)
    raise ValueError('unknown dataset %s' % name)
