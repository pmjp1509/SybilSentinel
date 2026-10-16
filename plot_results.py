#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read results/*.json and produce plots + a markdown summary table.

Outputs (in results/): acc_vs_rounds.png, acc_vs_sybil.png, detection.png, summary.md
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_NO, C_DEF, C_BASE = '#d1495b', '#2e86ab', '#4a4a4a'


def load(out_dir, iid):
    runs = {}
    for f in glob.glob(os.path.join(HERE, out_dir, '*.json')):
        r = json.load(open(f))
        c = r['config']
        if int(c['iid']) != iid:
            continue
        defense = c.get('defense', 'filter' if c.get('enable_filter') else 'none')
        if defense not in ('none', 'filter'):          # other defenses are plotted by plot_compare.py
            continue
        if int(c['sybil']) > 0 and c.get('attack', 'labelflip') != 'labelflip':
            continue
        key = (int(c['sybil']), defense == 'filter')
        runs.setdefault(key, []).append(r)
    return runs


def mean_curve(rs, field='acc'):
    return np.mean([r[field] for r in rs], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results')
    p.add_argument('--iid', type=int, default=1)
    a = p.parse_args()
    runs = load(a.out_dir, a.iid)
    if not runs:
        raise SystemExit('no result files found in %s' % a.out_dir)
    sybils = sorted({k[0] for k in runs})
    out = os.path.join(HERE, a.out_dir)
    tag = 'IID' if a.iid else 'non-IID'
    n_clients = next(iter(runs.values()))[0]['config']['num_users']

    # 1) accuracy vs rounds -----------------------------------------------------------
    fig, axes = plt.subplots(1, len(sybils), figsize=(4.6 * len(sybils), 3.8), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sybils):
        for filt, col, lab in ((False, C_NO, 'FedAvg + DP (no defense)'), (True, C_DEF, 'FedAvg + DP + similarity filter')):
            if (s, filt) in runs:
                ax.plot(mean_curve(runs[(s, filt)]), color=col, lw=1.8, label=lab)
        ax.set_title('{}% Sybil clients'.format(int(round(100.0 * s / n_clients))))
        ax.set_xlabel('communication round')
        ax.grid(alpha=.25)
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel('global test accuracy (%)')
    axes[0][0].legend(loc='lower right', fontsize=8)
    fig.suptitle('MNIST, {}: accuracy per round'.format(tag))
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'acc_vs_rounds.png'), dpi=150)
    plt.close(fig)

    # 2) final accuracy vs % sybils (bar chart) ---------------------------------------
    def final(rs):
        return np.mean([r['summary']['mean_last5_acc'] for r in rs])

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(sybils))
    w = 0.38
    v_no = [final(runs[(s, False)]) if (s, False) in runs else np.nan for s in sybils]
    v_def = [final(runs[(s, True)]) if (s, True) in runs else np.nan for s in sybils]
    b1 = ax.bar(x - w / 2, v_no, w, color=C_NO, label='no defense')
    b2 = ax.bar(x + w / 2, v_def, w, color=C_DEF, label='similarity filter')
    for b in list(b1) + list(b2):
        if not np.isnan(b.get_height()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, '%.1f' % b.get_height(), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['{}%'.format(int(round(100.0 * s / n_clients))) for s in sybils])
    ax.set_xlabel('share of Sybil clients')
    ax.set_ylabel('test accuracy, mean of last 5 rounds (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title('MNIST, {}: accuracy vs. Sybil share'.format(tag))
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'acc_vs_sybil.png'), dpi=150)
    plt.close(fig)

    # 3) detection: TPR / FPR of the filter --------------------------------------------
    det_s = [s for s in sybils if (s, True) in runs]
    tprs, fprs = [], []
    for s in det_s:
        sm = [r['summary'] for r in runs[(s, True)]]
        tp, fn = sum(m['tp'] for m in sm), sum(m['fn'] for m in sm)
        fp, tn = sum(m['fp'] for m in sm), sum(m['tn'] for m in sm)
        tprs.append(tp / (tp + fn) if tp + fn else np.nan)
        fprs.append(fp / (fp + tn) if fp + tn else np.nan)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(det_s))
    ax.bar(x - w / 2, [100 * t for t in tprs], w, color=C_DEF, label='TPR (sybils caught)')
    ax.bar(x + w / 2, [100 * f for f in fprs], w, color=C_BASE, label='FPR (honest wrongly dropped)')
    for i, (t, f) in enumerate(zip(tprs, fprs)):
        if not np.isnan(t):
            ax.text(i - w / 2, 100 * t + 1.5, '%.0f%%' % (100 * t), ha='center', fontsize=8)
        if not np.isnan(f):
            ax.text(i + w / 2, 100 * f + 1.5, '%.1f%%' % (100 * f), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['{}%'.format(int(round(100.0 * s / n_clients))) for s in det_s])
    ax.set_xlabel('share of Sybil clients')
    ax.set_ylabel('%')
    ax.set_ylim(0, 112)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Filter detection performance (per sampled client)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'detection.png'), dpi=150)
    plt.close(fig)

    # 4) markdown table ------------------------------------------------------------------
    base = final(runs[(0, False)]) if (0, False) in runs else np.nan
    lines = ['# Results summary (MNIST, {}, {} clients)\n'.format(tag, n_clients),
             '| Sybil % | Defense | Final acc (last-5 mean) | Drop vs. clean baseline | Label-flip rate | TPR | FPR |',
             '|---|---|---|---|---|---|---|']
    for s in sybils:
        for filt in (False, True):
            if (s, filt) not in runs:
                continue
            rs = runs[(s, filt)]
            acc = final(rs)
            flip = np.mean([r['summary']['final_flip_rate'] for r in rs])
            sm = [r['summary'] for r in rs]
            tp, fn = sum(m['tp'] for m in sm), sum(m['fn'] for m in sm)
            fp, tn = sum(m['fp'] for m in sm), sum(m['tn'] for m in sm)
            tpr = '%.0f%%' % (100.0 * tp / (tp + fn)) if (filt and tp + fn) else '-'
            fpr = '%.1f%%' % (100.0 * fp / (fp + tn)) if (filt and fp + tn) else '-'
            lines.append('| {}% | {} | {:.2f}% | {:+.2f} pts | {:.1f}% | {} | {} |'.format(
                int(round(100.0 * s / n_clients)), 'filter' if filt else 'none', acc, acc - base, flip, tpr, fpr))
    lines.append('\nTPR = fraction of Sybil submissions the filter dropped; FPR = fraction of honest submissions dropped.')
    lines.append('Label-flip rate = % of test images classified as the attacker\'s target label (9 - y).')
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nSaved plots + summary.md to', out)


if __name__ == '__main__':
    main()
