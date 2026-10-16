#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Well-known robust aggregation rules, used as COMPARISON baselines for our similarity filter.

All functions take the list of client weight dicts `updates` and the current global weights `reference`
and return the new global weights.

  krum / multi-krum : Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant GD", NeurIPS 2017
  median            : coordinate-wise median,        Yin et al., "Byzantine-Robust Distributed Learning", ICML 2018
  trimmed_mean      : coordinate-wise trimmed mean,  Yin et al., ICML 2018
  foolsgold         : similarity-based soft re-weighting, Fung et al., "Mitigating Sybils in Federated Learning
                      Poisoning", 2018/RAID 2020. NOTE: this is a *stateless, per-round* variant. The original keeps a
                      history of every client's accumulated updates; in our simulation clients are sampled at random
                      so a client rarely has history, hence we use the current round only.
"""

import copy
from typing import Dict, List

import numpy as np
import torch

from defenses.sybil_filter import cosine_similarity_matrix, flatten_update

Weights = Dict[str, torch.Tensor]


def _float_keys(w):
    return [k for k, v in w.items() if torch.is_floating_point(v)]


def _from_vectors(base: Weights, vec: torch.Tensor) -> Weights:
    """Write a flat vector back into a copy of the state dict `base` (float tensors only)."""
    out = copy.deepcopy(base)
    i = 0
    for k in _float_keys(base):
        n = base[k].numel()
        out[k] = vec[i:i + n].reshape(base[k].shape).to(base[k].dtype)
        i += n
    return out


def _stack(updates, reference):
    return torch.stack([flatten_update(u, reference) for u in updates])      # (N, D) updates (w_i - w_ref)


def _apply(reference: Weights, delta: torch.Tensor) -> Weights:
    ref_vec = flatten_update(reference)
    return _from_vectors(reference, ref_vec + delta)


def krum_select(updates: List[Weights], reference: Weights, f: int, m: int = 1):
    """Return indices of the m updates with the smallest Krum score (sum of squared distances to the
    n-f-2 nearest other updates). m=1 is Krum, m>1 is Multi-Krum."""
    n = len(updates)
    if n <= 2:
        return list(range(n))
    X = _stack(updates, reference)
    d2 = torch.cdist(X, X) ** 2
    k = max(n - f - 2, 1)
    scores = []
    for i in range(n):
        others = torch.cat([d2[i, :i], d2[i, i + 1:]])
        scores.append(torch.sort(others)[0][:k].sum().item())
    return list(np.argsort(scores)[:max(min(m, n), 1)])


def krum(updates, reference, f=None, multi=True):
    """(Multi-)Krum. f = assumed number of attackers (default: 30% of clients, limited to (n-3)//2)."""
    n = len(updates)
    if f is None:
        f = min(int(0.3 * n), max((n - 3) // 2, 0))
    m = max(n - f, 1) if multi else 1
    idx = krum_select(updates, reference, f, m)
    X = _stack(updates, reference)[idx]
    return _apply(reference, X.mean(0)), idx


def median(updates, reference):
    X = _stack(updates, reference)
    return _apply(reference, X.median(dim=0).values if len(X) % 2 else X.sort(0).values[len(X) // 2 - 1:len(X) // 2 + 1].mean(0))


def trimmed_mean(updates, reference, trim_ratio=0.2):
    X = _stack(updates, reference)
    n = len(X)
    t = min(int(trim_ratio * n), (n - 1) // 2)
    S = X.sort(dim=0).values
    return _apply(reference, S[t:n - t].mean(0))


def foolsgold_weights(updates, reference, kappa=1.0):
    """FoolsGold weighting (Fung et al.): weight_i = logit-rescaled (1 - max_j cos_ij) with pardoning."""
    n = len(updates)
    if n < 2:
        return np.ones(n)
    X = _stack(updates, reference)
    cs = cosine_similarity_matrix(X).numpy().astype(np.float64)
    np.fill_diagonal(cs, -np.inf)
    maxcs = cs.max(axis=1)
    # pardoning: do not punish honest clients that merely resemble a more "similar" one
    for i in range(n):
        for j in range(n):
            if i != j and maxcs[i] < maxcs[j] and np.isfinite(cs[i, j]):
                cs[i, j] = cs[i, j] * maxcs[i] / maxcs[j]
    np.fill_diagonal(cs, -np.inf)
    wv = 1.0 - cs.max(axis=1)
    wv = np.clip(wv, 0.0, 1.0)
    top = wv.max()
    wv = wv / top if top > 0 else np.ones(n)
    wv[wv == 1.0] = 0.99
    wv = kappa * (np.log(wv / (1.0 - wv)) + 0.5)
    wv[np.isinf(wv) & (wv > 0)] = 1.0
    wv = np.clip(wv, 0.0, 1.0)
    if wv.sum() <= 1e-12:                       # everybody looks like a sybil -> fall back to uniform
        wv = np.ones(n)
    return wv


def foolsgold(updates, reference):
    wv = foolsgold_weights(updates, reference)
    X = _stack(updates, reference)
    w = torch.tensor(wv / wv.sum(), dtype=X.dtype).unsqueeze(1)
    return _apply(reference, (X * w).sum(0)), wv


DEFENSES = ['none', 'filter', 'filter_median', 'krum', 'median', 'trimmed', 'foolsgold']
