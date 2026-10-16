#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local client training. Works on in-memory tensors (much faster than a DataLoader on CPU)."""

import numpy as np
import torch
from torch import nn

from utils.backdoor import add_trigger


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, flip_labels=False, backdoor=False):
        """flip_labels=True -> label-flipping poison: y -> (num_classes - 1 - y).
        backdoor=True      -> a fraction (--backdoor_frac) of the samples get a trigger patch and the
                              label --backdoor_target; the rest stay clean so the model still works normally."""
        self.args = args
        idxs = torch.as_tensor(np.asarray(list(idxs), dtype=np.int64))
        x = dataset.x[idxs]
        y = dataset.y[idxs]
        if flip_labels:
            y = (args.num_classes - 1) - y
        if backdoor:
            n_poison = int(round(args.backdoor_frac * len(y)))
            sel = torch.randperm(len(y))[:n_poison]
            x = x.clone()
            y = y.clone()
            x[sel] = add_trigger(x[sel], dataset.x.max().item())
            y[sel] = args.backdoor_target
        self.x = x.to(args.device)
        self.y = y.to(args.device)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        n = len(self.y)
        bs = self.args.local_bs
        epoch_loss = []
        for _ in range(self.args.local_ep):
            perm = torch.randperm(n, device=self.x.device)
            batch_loss = []
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                net.zero_grad()
                loss = self.loss_func(net(self.x[b]), self.y[b])
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
