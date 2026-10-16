#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sybil attack strategies.

One adversary controls `--sybil` fake clients (fixed ids for the whole run). Being ONE adversary, it needs to
compute a malicious update only once per round and submits (slightly perturbed) copies of it from all its
Sybil identities. The perturbation (`--sybil_jitter`) controls how well the Sybils try to hide:
cos(sybil_i, sybil_j) ~ 1 / (1 + jitter^2).

  labelflip : train on a shared shard with flipped labels (y -> 9-y), then boost the update by `sybil_boost`
  signflip  : train honestly on the shared shard, then send the NEGATED, boosted update
  random    : one shared random update per round
  backdoor  : train on the shared shard where a fraction of images carry a trigger patch labelled `backdoor_target`
              (the rest are clean, so overall accuracy stays high), then boost ("model replacement",
              Bagdasaryan et al., AISTATS 2020). Success is measured by the backdoor rate, not by accuracy.
  adaptive  : not a separate mode -- an evasion setting: raise --sybil_jitter (and/or use --sybil_dp) so that the
              Sybils are no longer similar to each other and slip under a fixed similarity threshold.
  dpnoise   : attack of the base repo -- each Sybil trains on its own shard but uses a tiny DP epsilon
              (= enormous independent Laplace noise). These updates are NOT similar to each other.

`--sybil_dp` makes the Sybils additionally add honest-level Laplace DP noise (stealthy variant).
"""

import copy

import numpy as np
import torch

from models.Update import LocalUpdate
from utils.dp import add_laplace_noise


def choose_sybil_ids(num_users, num_sybil, seed):
    """Fixed set of Sybil client ids (independent of the training RNG)."""
    rng = np.random.RandomState(seed + 12345)
    return set(int(i) for i in rng.choice(num_users, num_sybil, replace=False)) if num_sybil > 0 else set()


def _train(args, net_glob, dataset, idxs, flip, backdoor=False):
    local = LocalUpdate(args=args, dataset=dataset, idxs=idxs, flip_labels=flip, backdoor=backdoor)
    w, loss = local.train(copy.deepcopy(net_glob).to(args.device))
    return {k: v.detach().clone() for k, v in w.items()}, loss


def _scaled(w, w_prev, factor):
    """w_prev + factor * (w - w_prev)"""
    return {k: (w_prev[k] + factor * (w[k] - w_prev[k])) if torch.is_floating_point(w[k]) else w[k] for k in w}


def _jitter(w, w_prev, rel):
    """Add per-tensor Gaussian noise with std = rel * RMS(update of that tensor)."""
    if rel <= 0:
        return w
    out = {}
    for k, v in w.items():
        if not torch.is_floating_point(v):
            out[k] = v
            continue
        d = v - w_prev[k]
        rms = d.pow(2).mean().sqrt()
        out[k] = v + rel * rms * torch.randn_like(v)
    return out


class SybilAdversary(object):
    def __init__(self, args):
        self.args = args
        self._round = None
        self._base = None

    def _base_update(self, net_glob, w_prev, dataset, adv_idxs, rnd):
        a = self.args
        if a.attack == 'labelflip':
            w, _ = _train(a, net_glob, dataset, adv_idxs, flip=True)
            return _scaled(w, w_prev, a.sybil_boost)
        if a.attack == 'backdoor':                       # trigger patch -> target label, boosted ("model replacement")
            w, _ = _train(a, net_glob, dataset, adv_idxs, flip=False, backdoor=True)
            return _scaled(w, w_prev, a.sybil_boost)
        if a.attack == 'signflip':
            w, _ = _train(a, net_glob, dataset, adv_idxs, flip=False)
            return _scaled(w, w_prev, -a.sybil_boost)
        if a.attack == 'random':
            g = torch.Generator().manual_seed(a.seed * 100003 + rnd)
            return {k: (v + 0.01 * a.sybil_boost * torch.randn(v.shape, generator=g).to(v.device))
                    if torch.is_floating_point(v) else v for k, v in w_prev.items()}
        raise ValueError(a.attack)

    def craft(self, net_glob, w_prev, dataset, own_idxs, adv_idxs, rnd):
        """Weights submitted by one Sybil client in round `rnd`."""
        a = self.args
        if a.attack == 'dpnoise':                       # base-repo attack: independent, very noisy, own data
            w, _ = _train(a, net_glob, dataset, own_idxs, flip=False)
            return add_laplace_noise(w, a.epochs, a.epsilon_sybil)
        if self._round != rnd:                          # compute the shared malicious update once per round
            self._base = self._base_update(net_glob, w_prev, dataset, adv_idxs, rnd)
            self._round = rnd
        w = _jitter(self._base, w_prev, a.sybil_jitter)
        if a.sybil_dp:                                  # stealthy variant: also look like an honest DP client
            w = add_laplace_noise(w, a.epochs, a.epsilon)
        return w
