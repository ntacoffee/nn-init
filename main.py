#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 後のコードを走らせるための共通の初期化
import numpy as np
import torch
import torch.nn as nn

import util

n_layers = 10
in_features = 1000
out_features = 1000

activation_sigmoid = nn.Sigmoid()
activation_tanh = nn.Tanh()
activation_relu = nn.ReLU()


func_init_weight = lambda weight: (
    nn.init.uniform_(weight, -1.0 / np.sqrt(in_features), 1.0 / np.sqrt(in_features))
)
net = util.CustumInitNet(
    n_layers=n_layers,
    in_features=in_features,
    out_features=out_features,
    activation=activation_sigmoid,
    func_init_weight=func_init_weight,
)
x_in = torch.empty((in_features,)).normal_(0, 1).requires_grad_()
util.visualize(net, x_in, hist_range=[0, 1])

func_init_weight = lambda weight: (
    nn.init.uniform_(weight, -1.0 / np.sqrt(in_features), 1.0 / np.sqrt(in_features))
)
net = util.CustumInitNet(
    n_layers=n_layers,
    in_features=in_features,
    out_features=out_features,
    activation=activation_tanh,
    func_init_weight=func_init_weight,
)
x_in = torch.empty((in_features,)).normal_(0, 1)
util.visualize(net, x_in, hist_range=[-1, 1])

func_init_weight = lambda weight: (
    nn.init.uniform_(weight, -1.0 / np.sqrt(in_features), 1.0 / np.sqrt(in_features))
)
net = util.CustumInitNet(
    n_layers=n_layers,
    in_features=in_features,
    out_features=out_features,
    activation=activation_relu,
    func_init_weight=func_init_weight,
)
x_in = torch.empty((in_features,)).normal_(0, 1)
util.visualize(net, x_in, hist_range=[0, 1])


func_init_weight = lambda weight: (
    nn.init.uniform_(
        weight,
        -np.sqrt(6 / (in_features + out_features)),
        np.sqrt(6 / (in_features + out_features)),
    )
)

net = util.CustumInitNet(
    n_layers=n_layers,
    in_features=in_features,
    out_features=out_features,
    activation=activation_relu,
    func_init_weight=func_init_weight,
)
x_in = torch.empty((in_features,)).normal_(0, 1)
util.visualize(net, x_in, hist_range=[0, 1])

func_init_weight = lambda weight: (
    nn.init.uniform_(
        weight,
        -np.sqrt((5 / 3) * 6 / (in_features + out_features)),
        np.sqrt((5 / 3) * 6 / (in_features + out_features)),
    )
)

net = util.CustumInitNet(
    n_layers=n_layers,
    in_features=in_features,
    out_features=out_features,
    activation=activation_tanh,
    func_init_weight=func_init_weight,
)
x_in = torch.empty((in_features,)).normal_(0, 1)
util.visualize(net, x_in, hist_range=[-1, 1])

func_init_weight = lambda weight: (
    nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
)
net = util.CustumInitNet(
    n_layers=n_layers,
    in_features=in_features,
    out_features=out_features,
    activation=activation_relu,
    func_init_weight=func_init_weight,
)
x_in = torch.empty((in_features,)).normal_(0, 1)
util.visualize(net, x_in, hist_range=[0, 1])
