#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plots + table for benchmark_compare.py. Reads results/compare/cmp_*.json.

Outputs in the same folder: compare_accuracy.png, compare_attack_success.png, compare_summary.md
Metrics are averaged over the last 10 rounds (single rounds are noisy under attack)."""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFENSES = ['none', 'filter', 'filter_median', 'krum', 'median', 'trimmed', 'foolsgold']
DEF_LABEL = {'none': 'no defense\n(FedAvg)', 'filter': 'similarity\nfilter (ours)', 'filter_median': 'filter +\nmedian (ours)', 'krum': 'Multi-Krum',
             'median': 'median', 'trimmed': 'trimmed\nmean', 'foolsgold': 'FoolsGold-\nstyle'}
ATTACKS = ['labelflip', 'backdoor', 'evasive']
ATK_LABEL = {'labelflip': 'label-flip (colluding)', 'backdoor': 'backdoor (colluding)',
             'evasive': 'label-flip, evasive (noisy Sybils)'}
ATK_COLOR = {'labelflip': '#2e86ab', 'backdoor': '#d1495b', 'evasive': '#edae49'}
LAST = 10


def load(d):
    out = {}
    for f in glob.glob(os.path.join(HERE, d, 'cmp_*.json')):
        r = json.load(open(f))
        out[r['name'][4:]] = r
    return out


def m(r, field):
    return float(np.mean(r[field][-LAST:]))


def attack_metric(atk):
    return 'backdoor_rate' if atk == 'backdoor' else 'flip_rate'


def grouped_bars(ax, data, ylabel, ylim, clean=None, annotate=True):
    atks = [a for a in ATTACKS if any(('%s_%s' % (a, d)) in data for d in DEFENSES)]
    x = np.arange(len(DEFENSES))
    w = 0.8 / max(len(atks), 1)
    for i, atk in enumerate(atks):
        vals = []
        for d in DEFENSES:
            r = data.get('%s_%s' % (atk, d))
            vals.append(np.nan if r is None else m(r, 'acc' if ylabel.startswith('test acc') else attack_metric(atk)))
        bars = ax.bar(x + (i - (len(atks) - 1) / 2) * w, vals, w * 0.92, color=ATK_COLOR[atk], label=ATK_LABEL[atk])
        if annotate:
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, '%.0f' % v, ha='center', fontsize=7)
    if clean is not None:
        ax.axhline(clean, color='#4a4a4a', ls='--', lw=1, label='clean run, no attack (%.1f%%)' % clean)
    ax.set_xticks(x)
    ax.set_xticklabels([DEF_LABEL[d] for d in DEFENSES], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/compare')
    a = p.parse_args()
    data = load(a.out_dir)
    if not data:
        raise SystemExit('no cmp_*.json found in %s' % a.out_dir)
    out = os.path.join(HERE, a.out_dir)
    clean = m(data['clean'], 'acc') if 'clean' in data else None
    n_syb = next(iter(data.values()))['config']['sybil']
    rounds = next(iter(data.values()))['config']['epochs']

    fig, ax = plt.subplots(figsize=(10, 4.2))
    grouped_bars(ax, data, 'test accuracy (%%), mean of last %d rounds' % LAST, (0, 125), clean)
    ax.legend(loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    ax.set_title('Accuracy under attack (%d%% Sybils, %d rounds): higher is better' % (n_syb, rounds), pad=24)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'compare_accuracy.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    grouped_bars(ax, data, 'attack success (%%), mean of last %d rounds' % LAST, (0, 125))
    ax.legend(loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    ax.set_title('Attack success (label-flip rate / backdoor rate): lower is better', pad=24)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'compare_attack_success.png'), dpi=150)
    plt.close(fig)

    lines = ['# Defense x attack comparison (MNIST, IID, 100 clients, %d%% Sybils, %d rounds, last-%d-round mean)\n'
             % (n_syb, rounds, LAST)]
    if clean is not None:
        lines.append('Clean run (no attack, no defense): **%.2f%%** accuracy.\n' % clean)
    lines.append('Each cell: accuracy % / attack-success % (label-flip rate, or backdoor rate for the backdoor attack).\n')
    lines.append('| Defense | ' + ' | '.join(ATK_LABEL[x] for x in ATTACKS) + ' |')
    lines.append('|---|' + '---|' * len(ATTACKS))
    for d in DEFENSES:
        row = []
        for atk in ATTACKS:
            r = data.get('%s_%s' % (atk, d))
            row.append('-' if r is None else '%.1f / %.1f' % (m(r, 'acc'), m(r, attack_metric(atk))))
        lines.append('| %s | %s |' % (DEF_LABEL[d].replace('\n', ' '), ' | '.join(row)))
    with open(os.path.join(out, 'compare_summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
