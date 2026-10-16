#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Laplace-noise 'differential privacy' exactly as done in the base repo
(ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL): noise scale b = 2 * rounds * max|w| / (600 * epsilon),
added to every weight tensor of a client's model."""

import torch


def add_laplace_noise(w, rounds, epsilon):
    out = {}
    for key, val in w.items():
        if not torch.is_floating_point(val):
            out[key] = val
            continue
        b = 2 * rounds * torch.max(torch.abs(val)) / (600 * epsilon)
        b = torch.clamp(b, min=1e-12)
        noise = torch.distributions.laplace.Laplace(torch.zeros(()), b.detach().cpu()).sample(val.size())
        out[key] = val + noise.to(val.device)
    return out
