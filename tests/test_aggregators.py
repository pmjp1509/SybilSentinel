#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for the comparison defenses. Run:  python tests/test_aggregators.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch

from defenses import robust_agg

DIM = 200


def scene(n_honest=7, n_bad=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    ref = {'w': torch.zeros(DIM), 'b': torch.zeros(5)}
    true_dir = torch.randn(DIM, generator=g)
    honest = [{'w': 0.5 * true_dir + 0.3 * torch.randn(DIM, generator=g), 'b': torch.randn(5, generator=g) * 0.1}
              for _ in range(n_honest)]
    evil = 20.0 * torch.randn(DIM, generator=g)
    bad = [{'w': evil + 0.05 * torch.randn(DIM, generator=g), 'b': torch.ones(5) * 50} for _ in range(n_bad)]
    return ref, honest + bad, true_dir, np.array([False] * n_honest + [True] * n_bad)


def err(w, true_dir):
    return (w['w'] - 0.5 * true_dir).norm().item()


def test_fedavg_is_ruined_but_robust_rules_are_not():
    ref, ups, td, _ = scene()
    from models.Fed import FedAvg
    base = err(FedAvg(ups), td)
    for fn in (lambda: robust_agg.median(ups, ref), lambda: robust_agg.trimmed_mean(ups, ref, trim_ratio=0.3),
               lambda: robust_agg.krum(ups, ref)[0], lambda: robust_agg.foolsgold(ups, ref)[0]):
        assert err(fn(), td) < base / 3
    # property of the method: the default 20% trim cannot fully remove a 30% attacker group, but still helps
    assert err(robust_agg.trimmed_mean(ups, ref), td) < base


def test_krum_selects_honest():
    ref, ups, _, is_bad = scene()
    idx = robust_agg.krum_select(ups, ref, f=3, m=1)
    assert not is_bad[idx[0]]
    idx = robust_agg.krum_select(ups, ref, f=3, m=5)
    assert not is_bad[idx].any()


def test_foolsgold_downweights_colluders():
    ref, ups, _, is_bad = scene()
    wv = robust_agg.foolsgold_weights(ups, ref)
    assert wv[is_bad].max() < wv[~is_bad].min()


def test_shapes_and_keys_preserved():
    ref, ups, _, _ = scene()
    for w in (robust_agg.median(ups, ref), robust_agg.trimmed_mean(ups, ref), robust_agg.krum(ups, ref)[0],
              robust_agg.foolsgold(ups, ref)[0]):
        assert set(w.keys()) == set(ref.keys())
        assert w['w'].shape == ref['w'].shape and w['b'].shape == ref['b'].shape


def test_median_even_count():
    ref = {'w': torch.zeros(1)}
    ups = [{'w': torch.tensor([v])} for v in (1.0, 2.0, 3.0, 100.0)]
    assert abs(robust_agg.median(ups, ref)['w'].item() - 2.5) < 1e-6


def test_tiny_inputs_do_not_crash():
    ref = {'w': torch.zeros(3)}
    one = [{'w': torch.ones(3)}]
    two = one + [{'w': 2 * torch.ones(3)}]
    for ups in (one, two):
        robust_agg.median(ups, ref)
        robust_agg.trimmed_mean(ups, ref)
        robust_agg.krum(ups, ref)
        robust_agg.foolsgold(ups, ref)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
