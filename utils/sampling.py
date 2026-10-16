#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split a dataset over clients (IID and non-IID)."""

import numpy as np


def iid_split(labels, num_users, rng):
    """Every client gets an equal random slice. Returns {client: np.ndarray of indices}."""
    n = len(labels)
    per = n // num_users
    perm = rng.permutation(n)
    return {i: perm[i * per:(i + 1) * per] for i in range(num_users)}


def noniid_split(labels, num_users, rng, shards_per_user=2):
    """Classic FedAvg pathological non-IID split: sort by label, cut in shards, give each client
    `shards_per_user` shards (so each client sees ~2 classes)."""
    labels = np.asarray(labels)
    num_shards = num_users * shards_per_user
    shard_size = len(labels) // num_shards
    idxs = np.argsort(labels, kind='stable')
    shard_ids = rng.permutation(num_shards)
    dict_users = {}
    for i in range(num_users):
        mine = shard_ids[i * shards_per_user:(i + 1) * shards_per_user]
        dict_users[i] = np.concatenate([idxs[s * shard_size:(s + 1) * shard_size] for s in mine])
    return dict_users


# names kept from the base repo
def mnist_iid(dataset, num_users, rng=None):
    return iid_split(dataset.targets, num_users, rng or np.random)


def mnist_noniid(dataset, num_users, rng=None):
    return noniid_split(dataset.targets, num_users, rng or np.random)


def cifar_iid(dataset, num_users, rng=None):
    return iid_split(dataset.targets, num_users, rng or np.random)
