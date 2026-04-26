# deepcleanv3
Machine learning experiments for gravitational-wave detector noise subtraction, building off of the original DeepClean model. 

This repository contains research code for training and evaluating neural network and transformer models that predict noise contributions in gravitational-wave strain data from auxiliary witness channels. The goal is to explore architectures that improve the scalability and flexibility of DeepClean-style denoising, especially when working with changing auxiliary channel sets or frequency bands.

## Overview

DeepClean is a machine learning framework for subtracting instrumental and environmental noise from gravitational-wave detector data. This repository builds on that idea by experimenting with:

- per-channel convolutional feature extraction
- channel aggregation strategies
- transformer-based cross-channel interaction modeling

The main training entry point is:

```bash
python dc-transform-train.py
