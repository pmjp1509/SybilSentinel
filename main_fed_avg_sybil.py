#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FedAvg with DP noise, a Sybil attack, and a selectable defense.

Modified from ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL (main_fed_avg_sybil.py).

Examples
  python main_fed_avg_sybil.py --sybil 0                                   # baseline, no attack
  python main_fed_avg_sybil.py --sybil 20                                  # attack, no defense
  python main_fed_avg_sybil.py --sybil 20 --enable_filter                  # attack + our similarity filter
  python main_fed_avg_sybil.py --sybil 20 --attack backdoor --defense krum # other attack / defense
"""

import copy
import json
import os
import time

import numpy as np
import torch

from attacks.sybil_attack import SybilAdversary, choose_sybil_ids
from defenses import robust_agg
from defenses.sybil_filter import filter_by_similarity
from models.Fed import FedAvg
from models.Nets import CNNCifar, CNNMnist, MLP
from models.test import test_img
from models.Update import LocalUpdate
from utils.data import load_dataset
from utils.dp import add_laplace_noise
from utils.options import args_parser
from utils.sampling import iid_split, noniid_split


def run_name(a):
    if a.run_name:
        return a.run_name
    return '{}_{}_{}_s{}_{}_{}_seed{}'.format(
        a.dataset, a.model, 'iid' if a.iid else 'noniid', a.sybil, a.attack if a.sybil else 'none', a.defense, a.seed)


def aggregate(args, w_locals, w_prev, is_sybil):
    """Apply the selected defense. Returns (new_global_weights, flagged_mask or None).
    flagged_mask[i] = True if client i was dropped as a suspected Sybil (only meaningful for 'filter')."""
    n = len(w_locals)
    if args.defense == 'filter':
        _, kept, scores = filter_by_similarity(w_locals, threshold=args.filter_threshold, mode=args.filter_mode,
                                               reference=w_prev, return_info=True)
        return FedAvg([w for w, k in zip(w_locals, kept) if k]), ~kept, scores
    if args.defense == 'filter_median':                      # hybrid: similarity filter, then coordinate-wise median
        _, kept, scores = filter_by_similarity(w_locals, threshold=args.filter_threshold, mode=args.filter_mode,
                                               reference=w_prev, return_info=True)
        return robust_agg.median([w for w, k in zip(w_locals, kept) if k], w_prev), ~kept, scores
    if args.defense == 'krum':
        w, _ = robust_agg.krum(w_locals, w_prev)
        return w, None, None
    if args.defense == 'median':
        return robust_agg.median(w_locals, w_prev), None, None
    if args.defense == 'trimmed':
        return robust_agg.trimmed_mean(w_locals, w_prev), None, None
    if args.defense == 'foolsgold':
        w, _ = robust_agg.foolsgold(w_locals, w_prev)
        return w, None, None
    return FedAvg(w_locals), None, None                     # none


def main(argv=None, progress_cb=None):
    """Run one experiment. `progress_cb(dict)` (optional) is called after every round (used by the Streamlit app)."""
    args = args_parser(argv)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # ---------------- data ----------------
    dataset_train, dataset_test = load_dataset(args.dataset, args.data_dir)
    args.num_channels = dataset_train.x.shape[1]
    if args.iid:
        dict_users = iid_split(dataset_train.targets, args.num_users, rng)
    elif args.dataset == 'mnist':
        dict_users = noniid_split(dataset_train.targets, args.num_users, rng)
    else:
        raise SystemExit('Error: only the IID setting is supported for CIFAR-10')

    # ---------------- model ----------------
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args)
    elif args.model == 'cnn':
        net_glob = CNNMnist(args)
    else:
        net_glob = MLP(int(np.prod(dataset_train.x.shape[1:])), 64, args.num_classes)
    net_glob = net_glob.to(args.device)
    net_glob.train()

    # ---------------- sybils ----------------
    sybil_ids = choose_sybil_ids(args.num_users, args.sybil, args.seed)
    adv_idxs = dict_users[min(sybil_ids)] if sybil_ids else None     # the adversary's single shared data shard
    adversary = SybilAdversary(args)
    m = max(int(args.frac * args.num_users), 1)

    name = run_name(args)
    print('Run: {} | device={} | clients/round={} | sybils in population={} ({:.0f}%) | defense={}'.format(
        name, args.device, m, len(sybil_ids), 100.0 * len(sybil_ids) / args.num_users, args.defense), flush=True)

    acc_list, loss_list, flip_list, bd_list, rounds_log = [], [], [], [], []
    tp = fn = fp = tn = 0
    t0 = time.time()

    for rnd in range(args.epochs):
        w_prev = copy.deepcopy(net_glob.state_dict())
        idxs_users = rng.choice(args.num_users, m, replace=False)

        w_locals, is_sybil = [], []
        for idx in idxs_users:
            if int(idx) in sybil_ids:
                w = adversary.craft(net_glob, w_prev, dataset_train, dict_users[idx], adv_idxs, rnd)
                is_sybil.append(True)
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, _ = local.train(copy.deepcopy(net_glob).to(args.device))
                w = {k: v.detach().clone() for k, v in w.items()}
                if args.dp:
                    w = add_laplace_noise(w, args.epochs, args.epsilon)
                is_sybil.append(False)
            w_locals.append(w)
        is_sybil = np.array(is_sybil)

        # ---------------- defense + aggregation ----------------
        # similarity scores are always computed for logging (cheap), they only change the model for 'filter'
        _, _, log_scores = filter_by_similarity(w_locals, threshold=args.filter_threshold, mode=args.filter_mode,
                                                reference=w_prev, return_info=True)
        w_new, flagged, _ = aggregate(args, w_locals, w_prev, is_sybil)
        net_glob.load_state_dict(w_new)

        if flagged is None:
            flagged = np.zeros(len(w_locals), dtype=bool)
        if args.defense in ('filter', 'filter_median'):  # detection statistics only exist for a client-level filter
            tp += int((flagged & is_sybil).sum())
            fn += int((~flagged & is_sybil).sum())
            fp += int((flagged & ~is_sybil).sum())
            tn += int((~flagged & ~is_sybil).sum())

        acc, tloss, flip, bd = test_img(net_glob, dataset_test, args)
        acc_list.append(acc)
        loss_list.append(tloss)
        flip_list.append(flip)
        bd_list.append(bd)
        info = {
            'round': rnd, 'acc': acc, 'loss': tloss, 'flip_rate': flip, 'backdoor_rate': bd,
            'n_sybil': int(is_sybil.sum()), 'n_honest': int((~is_sybil).sum()),
            'flagged_sybil': int((flagged & is_sybil).sum()), 'flagged_honest': int((flagged & ~is_sybil).sum()),
            'score_sybil_mean': float(log_scores[is_sybil].mean()) if is_sybil.any() else None,
            'score_honest_mean': float(log_scores[~is_sybil].mean()) if (~is_sybil).any() else None,
            'score_honest_max': float(log_scores[~is_sybil].max()) if (~is_sybil).any() else None,
        }
        rounds_log.append({k: v for k, v in info.items() if k not in ('acc', 'loss', 'flip_rate', 'backdoor_rate')}
                          | {'round': rnd})
        print('Round {:3d} | acc {:6.2f}% | flip {:5.2f}% | backdoor {:5.2f}% | sybils {} | flagged sybil/honest {}/{} | {:.0f}s'.format(
            rnd, acc, flip, bd, info['n_sybil'], info['flagged_sybil'], info['flagged_honest'], time.time() - t0),
            flush=True)
        if progress_cb is not None:
            info['seconds'] = time.time() - t0
            progress_cb(info)

    result = {
        'name': name,
        'config': {k: v for k, v in vars(args).items() if k != 'device'},
        'acc': acc_list, 'loss': loss_list, 'flip_rate': flip_list, 'backdoor_rate': bd_list, 'rounds': rounds_log,
        'summary': {
            'final_acc': acc_list[-1],
            'mean_last5_acc': float(np.mean(acc_list[-5:])),
            'best_acc': max(acc_list),
            'final_flip_rate': flip_list[-1],
            'final_backdoor_rate': bd_list[-1],
            'mean_last5_backdoor': float(np.mean(bd_list[-5:])),
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'tpr': tp / (tp + fn) if (tp + fn) else None,
            'fpr': fp / (fp + tn) if (fp + tn) else None,
            'seconds': time.time() - t0,
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, name + '.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=1)
    s = result['summary']
    print('Done. final acc {:.2f}% | TPR {} | FPR {} | saved {}'.format(s['final_acc'], s['tpr'], s['fpr'], path))
    return result


if __name__ == '__main__':
    main()
