#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class CustumInitNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        func_init_weight: Callable[[torch.Tensor], None],
    ) -> None:
        super().__init__()

        self._fc = []
        for i in range(n_layers):
            self._fc.append(nn.Linear(in_features, out_features))
            func_init_weight(self._fc[i].weight.data)
            nn.init.zeros_(self._fc[i].bias.data)

        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        self._out_list = []
        for i in range(len(self)):
            out = self._fc[i](out)
            out = self._activation(out)
            self._out_list.append(out)

        return out

    def get_layter_out_list(self) -> List[torch.Tensor]:
        return self._out_list

    def __len__(self):
        return len(self._fc)


def visualize(net: CustumInitNet, x_in: torch.Tensor, hist_range: List[int]) -> None:

    out = net.forward(x_in)
    x_out_list = net.get_layter_out_list()

    loss_func = nn.MSELoss()
    loss_value = loss_func(out, torch.zeros_like(out))
    loss_value.backward()

    n_graphs = len(net)
    n_square = np.ceil(np.sqrt(n_graphs))
    n_colum = n_square
    n_row = np.ceil(float(n_graphs) / n_colum)

    figure = plt.figure()
    for i in range(len(net)):
        ax = figure.add_subplot(n_row, n_colum, i + 1)
        ax.hist(x_out_list[i].detach().numpy(), bins=20, range=hist_range, density=True)
        ax.set_xlim
        ax.set_ylim([0, 1])
    plt.show()

    figure = plt.figure()
    for i in range(len(net)):
        ax = figure.add_subplot(n_row, n_colum, i + 1)
        grad = np.reshape(net._fc[i].weight.grad.detach().numpy(), (-1,)) * 1000
        ax.hist(
            np.random.choice(grad, size=1000, replace=False), bins=20, density=True,
        )
        ax.set_xlim
    plt.show()
