#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defense x attack comparison: our similarity filter vs Krum / median / trimmed mean / FoolsGold.

  python benchmark_compare.py                 # 30 rounds, 30% Sybils  (~20 min on 2 CPU cores)
  python benchmark_compare.py --epochs 15     # quick

Attack columns:
  labelflip : colluding Sybils, label-flip + boost (Sybils ~96% similar to each other)
  backdoor  : colluding Sybils, trigger-patch backdoor + boost
  evasive   : label-flip Sybils that add large independent noise (--sybil_jitter 1.0) to hide from similarity checks
Results are written to results/compare/ and plotted by plot_compare.py.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ATTACKS = {
    'labelflip': ['--attack', 'labelflip'],
    'backdoor': ['--attack', 'backdoor'],
    'evasive': ['--attack', 'labelflip', '--sybil_jitter', '1.0'],
}
DEFENSES = ['none', 'filter', 'filter_median', 'krum', 'median', 'trimmed', 'foolsgold']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--sybil', type=int, default=30, help="number of Sybils among the 100 clients")
    p.add_argument('--attacks', nargs='+', default=list(ATTACKS), choices=list(ATTACKS))
    p.add_argument('--defenses', nargs='+', default=DEFENSES, choices=DEFENSES)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--out_dir', default='results/compare')
    p.add_argument('--no_plot', action='store_true')
    a = p.parse_args()

    jobs = []
    names = [('clean', 'none', ['--sybil', '0'])]          # clean reference run (no attack, no defense)
    for atk in a.attacks:
        for d in a.defenses:
            names.append(('{}_{}'.format(atk, d), d, ['--sybil', str(a.sybil)] + ATTACKS[atk]))
    for name, d, extra in names:
        if os.path.exists(os.path.join(HERE, a.out_dir, 'cmp_' + name + '.json')):
            print('skip (exists):', name)
            continue
        cmd = [sys.executable, os.path.join(HERE, 'main_fed_avg_sybil.py'), '--epochs', str(a.epochs),
               '--defense', d, '--seed', str(a.seed), '--num_threads', '1',
               '--out_dir', os.path.join(HERE, a.out_dir), '--run_name', 'cmp_' + name] + extra
        jobs.append((name, cmd))

    os.makedirs(os.path.join(HERE, 'log'), exist_ok=True)
    running, t0 = [], time.time()
    while jobs or running:
        while jobs and len(running) < a.jobs:
            name, cmd = jobs.pop(0)
            logf = open(os.path.join(HERE, 'log', 'cmp_' + name + '.log'), 'w')
            print('[{:5.0f}s] start {}'.format(time.time() - t0, name), flush=True)
            running.append((name, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)))
        for item in running[:]:
            if item[1].poll() is not None:
                print('[{:5.0f}s] done  {} (exit {})'.format(time.time() - t0, item[0], item[1].returncode), flush=True)
                running.remove(item)
        time.sleep(2)

    if not a.no_plot:
        subprocess.call([sys.executable, os.path.join(HERE, 'plot_compare.py'), '--out_dir', a.out_dir], cwd=HERE)


if __name__ == '__main__':
    main()
