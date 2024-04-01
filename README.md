# Natural Policy Gradient for Low Rank MDP with log-linear Parametrization

## 1. Introduction

To get an overview of this project, please refer to my [presentation](Experiments_with_Low_Rank_MDP.pdf). To check out the implementation details and the results of the codes, please refer to:

[Low_Rank_MDP_1_Dimensional_Feature.ipynb](Low_Rank_MDP_1_Dimensional_Feature.ipynb) (the environment has a low-rank structure and an one-dimensional feature)

[Low_Rank_MDP_S_A_Dimensional_Feature.ipynb](Low_Rank_MDP_S_A_Dimensional_Feature.ipynb) (the environment has a low-rank structure and an S $\times$ A dimensional feature. S is the number of possible states of the environment, and A is the number of possible actions that can be taken by the agent.)

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

## 2. Reference
