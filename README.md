# `CounterDKL`
[GPyTorch](https://gpytorch.ai/) implementation of Counterfactual Learning with Multioutput Deep Kernels, from [Caron et. al. (2022)](https://openreview.net/pdf?id=iGREAJdULX). The folders contain all necessary code to replicate examples and simulated studies in the original paper.

## Why Counterfactual Multitask Learning with Deep Kernels?

The code implements Counterfactual Multitask Learning via Gaussian Processes [Alaa & Van Der Schaar (2017)](https://proceedings.neurips.cc/paper/2017/file/6a508a60aa3bf9510ea6acb021c94b48-Paper.pdf) or via Deep Kernel Learning [Caron et. al. (2022)](https://openreview.net/pdf?id=iGREAJdULX), based on scalable package implementation in Torch.

DKL is particularly useful in policy learning (RL) settings where the action space is large (and possibly combinatorial), in addition to large input/context space and multiple outcomes.
