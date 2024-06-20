#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:37:02 2021

@author: root
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear


class PASGC(torch.nn.Module):
    def __init__(self, nfeature, nclass):
        super(PASGC, self).__init__()
        self.conv1 = Linear(nfeature, nclass, bias=True)

    def forward(self, data):
        x = data.x
        x = self.conv1(x)
        return F.log_softmax(x, dim=1)

