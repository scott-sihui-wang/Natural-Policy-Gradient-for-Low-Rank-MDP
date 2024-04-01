# Natural Policy Gradient for Low Rank MDP with log-linear Parametrization

## 1. Introduction

To get an overview of this project, please refer to my [presentation](Experiments_with_Low_Rank_MDP.pdf). To check out the implementation details and the results of the codes, please refer to:

[Low_Rank_MDP_1_Dimensional_Feature.ipynb](Low_Rank_MDP_1_Dimensional_Feature.ipynb) (the environment has a low-rank structure and an one-dimensional feature)

[Low_Rank_MDP_S_A_Dimensional_Feature.ipynb](Low_Rank_MDP_S_A_Dimensional_Feature.ipynb) (the environment has a low-rank structure and an S $\times$ A dimensional feature. S is the number of possible states of the environment, and A is the number of possible actions that can be taken by the agent)

You can also find the code on `Colab`:

[Low-Rank MDP, 1 Dimensional Feature](https://colab.research.google.com/drive/11fxNyIbSwCvVfEAFIP_tp2z0F3euM3i_?usp=sharing) 

[Low-Rank MDP, S * A Dimensional Feature](https://colab.research.google.com/drive/1ME6TF5S35SRgFqJ49ziV2r2ehAnmX8ZI?usp=sharing)

### 1.1 Background and Motivation

It was well-established that, for `tabular softmax parametrization`, `natural policy gradient` method can obtain `linear convergence rate` with `geometrically increasing step-size`.

Recent findings suggest that, for `log-linear policy parametrization`, `natural policy gradient` method can also obtain `linear rate` with `geometrically increasing step-size`, _given_ that the environment has a `low-rank structure`.

In this project, we will give an example of `low-rank MDP`, and test if `natural policy gradient` method can achieve `linear rate` in such settings.

### 1.2 Tasks of this Project

- Implement `MDP` with `low-rank structure`;
- Implement `NPG` with different step-size;
- Analyze the `convergence rate`.

**Topics:** _Reinforcement Learning_, _Natural Policy Gradient (NPG)_, _Markov Decision Process (MDP)_

**Skills:** _Python_, _JAX_, _Numpy_, _Jupyter Lab_, _Colab_

## 2. Implementation Details and Results:

This is the low-rank MDP environment we designed for experiments:

![low-rank MDP environment]()

This is how the `Q value` converges to the _optimal_ `Q value` by `NPG` iterations:

![NPG Convergence]()

## 3. Reference

[Linear Convergence for Natural Policy Gradient with Log-linear Policy Parametrization](https://arxiv.org/pdf/2209.15382.pdf)
