#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cosine-similarity Sybil filter (FoolsGold-style) to be combined with the DP noise of the base repo.

Idea: honest clients train on different data, so their updates are diverse. Sybils controlled by one
adversary push the same malicious direction, so their updates are (almost) parallel.

Steps (see README):
  1. vectorize   : flatten every client's update into one vector v_i
                   (if `reference` = current global weights is given, v_i = w_i - reference, i.e. the *update*;
                   this matters: raw weights of all clients are ~identical because they share the global model)
  2. cosine      : cos_ij = <v_i, v_j> / (||v_i|| ||v_j||)
  3. score       : mode='max'  -> S_i = max_{j != i} cos_ij      (FoolsGold-style, default)
                   mode='mean' -> S_i = 1/(N-1) * sum_{j != i} cos_ij   (literal version of the project plan)
  4. threshold   : S_i > tau  ->  suspected Sybil
  5. filter      : drop suspected updates
  6. aggregate   : caller runs FedAvg on the kept updates

Why 'max' is the default: with only a few Sybils among N clients, the *mean* similarity of a Sybil is
about (1 + (N-2) * cos_sybil_honest) / (N-1), far below 0.9, so a mean-based rule with tau = 0.9 can only
fire when almost every client colludes. The *max* similarity is ~1 for any pair of colluding Sybils.
"""

from typing import Dict, List

import numpy as np
import torch

Update = Dict[str, torch.Tensor]


def flatten_update(update: Update, reference: Update = None) -> torch.Tensor:
    """Flatten all floating-point tensors of a state_dict into one 1-D vector (optionally minus reference)."""
    parts = []
    for k, v in update.items():
        if not torch.is_floating_point(v):
            continue
        t = v.detach().float().cpu()
        if reference is not None:
            t = t - reference[k].detach().float().cpu()
        parts.append(t.flatten())
    return torch.cat(parts)


def cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """vectors: (N, D) -> (N, N) cosine similarity matrix."""
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = vectors / norms
    return unit @ unit.t()


def similarity_scores(updates: List[Update], mode: str = 'max', reference: Update = None) -> np.ndarray:
    """Suspicion score S_i for every client (higher = more similar to the others)."""
    n = len(updates)
    if n < 2:
        return np.zeros(n)
    vecs = torch.stack([flatten_update(u, reference) for u in updates])
    sim = cosine_similarity_matrix(vecs).numpy().astype(np.float64)
    np.fill_diagonal(sim, np.nan)
    if mode == 'max':
        return np.nanmax(sim, axis=1)
    if mode == 'mean':
        return np.nanmean(sim, axis=1)
    raise ValueError("mode must be 'max' or 'mean'")


def filter_by_similarity(updates: List[Update], threshold: float = 0.9, mode: str = 'max',
                         reference: Update = None, return_info: bool = False):
    """Remove updates whose similarity score exceeds `threshold`.

    Returns the filtered list of updates. With return_info=True returns (kept_updates, kept_mask, scores)
    where kept_mask[i] is True if update i was kept (so flagged = ~kept_mask)."""
    scores = similarity_scores(updates, mode=mode, reference=reference)
    kept = scores <= threshold
    if len(updates) and not kept.any():          # never drop everybody: keep the least suspicious client
        kept[int(np.argmin(scores))] = True
    filtered = [u for u, k in zip(updates, kept) if k]
    if return_info:
        return filtered, kept, scores
    return filtered
