#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for the similarity filter. Run:  python tests/test_filter.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch

from defenses.sybil_filter import filter_by_similarity, similarity_scores


def make_updates(n_honest=8, n_sybil=3, dim=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    common = torch.randn(dim, generator=g) * 0.3                       # weak shared signal among honest clients
    honest = [{'w': common + torch.randn(dim, generator=g)} for _ in range(n_honest)]
    bad = torch.randn(dim, generator=g)
    sybils = [{'w': bad + 0.05 * torch.randn(dim, generator=g)} for _ in range(n_sybil)]
    return honest + sybils, np.array([False] * n_honest + [True] * n_sybil)


def test_max_mode_flags_colluding_sybils():
    ups, is_syb = make_updates()
    kept_ups, kept, scores = filter_by_similarity(ups, threshold=0.9, mode='max', return_info=True)
    assert (~kept)[is_syb].all(), 'all sybils must be flagged'
    assert kept[~is_syb].all(), 'no honest client may be flagged'
    assert len(kept_ups) == (~is_syb).sum()


def test_mean_mode_is_too_weak_for_few_sybils():
    ups, is_syb = make_updates()
    _, kept, _ = filter_by_similarity(ups, threshold=0.9, mode='mean', return_info=True)
    assert kept.all(), 'literal mean rule with tau=0.9 does not fire when only 3 of 11 collude'


def test_reference_makes_updates_from_weights():
    ups, is_syb = make_updates()
    ref = {'w': torch.full((500,), 5.0)}                                # shared global model
    weights = [{'w': u['w'] + ref['w']} for u in ups]
    _, kept_ref, _ = filter_by_similarity(weights, 0.9, 'max', reference=ref, return_info=True)
    _, kept_raw, _ = filter_by_similarity(weights, 0.9, 'max', return_info=True)
    assert (~kept_ref)[is_syb].all() and kept_ref[~is_syb].all()
    assert (~kept_raw).all() or not kept_raw.all()                      # raw weights are all ~parallel -> useless


def test_never_drops_everyone():
    ups = [{'w': torch.ones(10)} for _ in range(4)]                     # identical -> all flagged
    out = filter_by_similarity(ups, 0.9, 'max')
    assert len(out) == 1


def test_single_update():
    assert len(filter_by_similarity([{'w': torch.ones(3)}], 0.9)) == 1
    assert similarity_scores([{'w': torch.ones(3)}]).tolist() == [0.0]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
