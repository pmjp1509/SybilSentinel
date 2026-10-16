#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line options for the Sybil-attack / defense FL experiments."""

import argparse


def args_parser(argv=None):
    p = argparse.ArgumentParser(description="Sybil attacks on Federated Learning: attack + similarity-filter defense")

    # ---- federated learning ----
    p.add_argument('--epochs', type=int, default=50, help="communication rounds")
    p.add_argument('--num_users', type=int, default=100, help="total number of clients K")
    p.add_argument('--frac', type=float, default=0.1, help="fraction of clients sampled each round C")
    p.add_argument('--local_ep', type=int, default=2, help="local epochs E")
    p.add_argument('--local_bs', type=int, default=10, help="local batch size B")
    p.add_argument('--bs', type=int, default=1000, help="test batch size")
    p.add_argument('--lr', type=float, default=0.01, help="learning rate")
    p.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")

    # ---- data / model ----
    p.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'])
    p.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'])
    p.add_argument('--iid', type=int, default=1, help="1 = IID split, 0 = non-IID (MNIST only)")
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--data_dir', type=str, default='data', help="where datasets are stored / downloaded")

    # ---- differential privacy (Laplace noise on honest clients, as in the base repo) ----
    p.add_argument('--dp', type=int, default=1, help="1 = honest clients add Laplace DP noise, 0 = no noise")
    p.add_argument('--epsilon', type=float, default=8.0, help="DP epsilon of honest clients")

    # ---- Sybil attack ----
    p.add_argument('--sybil', type=int, default=0,
                   help="number of Sybil clients among the --num_users clients (e.g. 20 of 100 = 20%%)")
    p.add_argument('--attack', type=str, default='labelflip',
                   choices=['labelflip', 'signflip', 'random', 'dpnoise', 'backdoor'],
                   help="labelflip: colluding sybils train on one shared label-flipped shard and boost the update; "
                        "signflip: colluding sybils send a negated, boosted update; "
                        "random: colluding sybils send one shared random update; "
                        "dpnoise: attack of the original base repo (sybils use a tiny epsilon -> huge independent noise)")
    p.add_argument('--sybil_boost', type=float, default=3.0, help="scaling factor applied to the malicious update")
    p.add_argument('--backdoor_target', type=int, default=0, help="label the backdoor trigger should cause")
    p.add_argument('--backdoor_frac', type=float, default=0.5,
                   help="fraction of the adversary's training images that carry the trigger (backdoor attack)")
    p.add_argument('--sybil_jitter', type=float, default=0.2,
                   help="relative per-client perturbation the Sybils add to hide (0 = identical copies). "
                        "cosine between two Sybils ~ 1/(1+jitter^2)")
    p.add_argument('--sybil_dp', action='store_true',
                   help="stealthy sybils: they also add the honest DP noise to their update (harder to detect)")
    p.add_argument('--epsilon_sybil', type=float, default=0.1, help="epsilon used by the 'dpnoise' attack")

    # ---- defense ----
    p.add_argument('--defense', type=str, default='none',
                   choices=['none', 'filter', 'filter_median', 'krum', 'median', 'trimmed', 'foolsgold'],
                   help="aggregation defense. filter = OUR cosine-similarity filter + FedAvg; filter_median = OUR hybrid (filter, then coordinate-wise median of the survivors); krum = Multi-Krum; "
                        "median / trimmed = coordinate-wise median / trimmed mean; foolsgold = FoolsGold-style weighting")
    p.add_argument('--enable_filter', action='store_true', help="shortcut for --defense filter")
    p.add_argument('--filter_threshold', type=float, default=0.9, help="similarity threshold tau")
    p.add_argument('--filter_mode', type=str, default='max', choices=['max', 'mean'],
                   help="max: flag client if its MAX cosine similarity to another client > tau (FoolsGold-style); "
                        "mean: flag client if its AVERAGE cosine similarity to the others > tau")

    # ---- misc ----
    p.add_argument('--gpu', type=int, default=-1, help="GPU id, -1 for CPU")
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num_threads', type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument('--out_dir', type=str, default='results', help="where the result JSON is written")
    p.add_argument('--run_name', type=str, default='', help="name of the result file (auto if empty)")
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args(argv)
    if a.enable_filter and a.defense == 'none':
        a.defense = 'filter'
    a.enable_filter = (a.defense == 'filter')
    return a
