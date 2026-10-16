#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from utils.backdoor import add_trigger


def test_img(net_g, datatest, args):
    """Returns (accuracy %, mean loss, flip_rate %, backdoor_rate %).

    flip_rate     = % of test images predicted as the label-flip target (num_classes-1-y), counted only where that
                    target differs from the true label. Measures a label-flip attack.
    backdoor_rate = % of triggered test images (true label != backdoor target) classified as the backdoor target.
                    Measures a backdoor attack (should be ~0-1% for a clean model)."""
    net_g.eval()
    loss_sum, correct, flipped, flip_total, bd_hit, bd_total = 0.0, 0, 0, 0, 0, 0
    n = len(datatest)
    trig_val = datatest.x.max().item()
    with torch.no_grad():
        for i in range(0, n, args.bs):
            x = datatest.x[i:i + args.bs].to(args.device)
            y = datatest.y[i:i + args.bs].to(args.device)
            out = net_g(x)
            loss_sum += F.cross_entropy(out, y, reduction='sum').item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            tgt = (args.num_classes - 1) - y
            mask = tgt != y
            flip_total += mask.sum().item()
            flipped += ((pred == tgt) & mask).sum().item()
            # backdoor success on triggered inputs
            m2 = y != args.backdoor_target
            if m2.any():
                pb = net_g(add_trigger(x[m2], trig_val)).argmax(1)
                bd_hit += (pb == args.backdoor_target).sum().item()
                bd_total += m2.sum().item()
    return (100.0 * correct / n, loss_sum / n, 100.0 * flipped / max(flip_total, 1),
            100.0 * bd_hit / max(bd_total, 1))
