#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backdoor trigger: a small bright square in the bottom-right corner of the image."""

import torch

PATCH = 4


def add_trigger(x, value):
    """x: (N, C, H, W). Returns a copy with a PATCH x PATCH square set to `value`."""
    x = x.clone()
    x[:, :, -PATCH - 1:-1, -PATCH - 1:-1] = value
    return x
