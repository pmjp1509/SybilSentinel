#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the whole benchmarking matrix (0/20/40 sybils x no-defense / filter) and then plot.

  python benchmark.py                       # 50 rounds, MNIST, IID  (about 20-30 min on a 2-core CPU)
  python benchmark.py --epochs 20 --jobs 2  # quicker
  python benchmark.py --iid 0               # non-IID MNIST
Extra flags after `--` are passed to main_fed_avg_sybil.py, e.g.  python benchmark.py -- --attack signflip
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--sybils', type=int, nargs='+', default=[0, 20, 40])
    p.add_argument('--iid', type=int, default=1)
    p.add_argument('--seeds', type=int, nargs='+', default=[1])
    p.add_argument('--jobs', type=int, default=2, help="parallel runs (1 CPU thread each)")
    p.add_argument('--out_dir', default='results')
    p.add_argument('--no_plot', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER, help="extra args for main_fed_avg_sybil.py (after --)")
    a = p.parse_args()
    extra = [x for x in a.extra if x != '--']

    tag = 'iid' if a.iid else 'noniid'
    cmds = []
    for seed in a.seeds:
        for s in a.sybils:
            for filt in (False, True):
                name = 'mnist_cnn_{}_s{}_{}_seed{}'.format(tag, s, 'filter' if filt else 'nofilter', seed)
                if os.path.exists(os.path.join(HERE, a.out_dir, name + '.json')):
                    print('skip (exists):', name)
                    continue
                cmd = [sys.executable, os.path.join(HERE, 'main_fed_avg_sybil.py'),
                       '--epochs', str(a.epochs), '--sybil', str(s), '--iid', str(a.iid), '--seed', str(seed),
                       '--num_threads', '1', '--out_dir', os.path.join(HERE, a.out_dir), '--run_name', name] + extra
                if filt:
                    cmd.append('--enable_filter')
                cmds.append((name, cmd))

    os.makedirs(os.path.join(HERE, 'log'), exist_ok=True)
    running, t0 = [], time.time()
    while cmds or running:
        while cmds and len(running) < a.jobs:
            name, cmd = cmds.pop(0)
            logf = open(os.path.join(HERE, 'log', name + '.log'), 'w')
            print('[{:5.0f}s] start {}'.format(time.time() - t0, name), flush=True)
            running.append((name, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)))
        for item in running[:]:
            if item[1].poll() is not None:
                print('[{:5.0f}s] done  {} (exit {})'.format(time.time() - t0, item[0], item[1].returncode), flush=True)
                running.remove(item)
        time.sleep(2)

    if not a.no_plot:
        subprocess.call([sys.executable, os.path.join(HERE, 'plot_results.py'), '--out_dir', a.out_dir, '--iid', str(a.iid)],
                        cwd=HERE)


if __name__ == '__main__':
    main()
