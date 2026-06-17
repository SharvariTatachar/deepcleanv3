# deepcleanv3
Machine learning experiments for gravitational-wave detector noise subtraction, building off of the original [DeepClean](https://github.com/ML4GW/deepcleanv2) model. 

This repository contains research code for training and evaluating neural network and transformer models that predict noise in gravitational-wave strain data from witness channels. The goal is to explore architectures that improve the scalability and flexibility of DeepClean-style denoising, especially when working with changing auxiliary channel sets, frequency bands, or cleaning problems.

## Overview

DeepClean is a machine learning framework for subtracting instrumental and environmental noise from gravitational-wave detector data. This repository builds on that idea by experimenting with:

- per-channel convolutional feature extraction
- channel aggregation strategies
- transformer-based cross-channel interaction modeling

The main training entry point for the new model is:

```bash
python3 dc-transform-train.py
```
The training entry point for the original DeepClean model is: 

```bash
python3 dc-train.py
```

## Repository Structure 
```
deepcleanv3/
│── channelconfigs           ← channel sets
│── configs                  ← training configs 
│── deepclean                ← DeepClean implementation
│── deepcleanhybrid          ← DeepClean-Transform implementation
│── figures                  ← experiment results, analysis, etc.
│── scripts                  ← scripts for analysis, dataset creation, etc.
│── dc-train.py              ← DeepClean training script
│── dc-transform-train.py    ← DeepClean-Transform training script
│── requirements.txt         ← virtual env requirements 

```
