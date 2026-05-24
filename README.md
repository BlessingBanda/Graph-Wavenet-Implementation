# STGNN-Implementations

Empirical investigation of graph construction strategies in Spatio Temporal Graph Neural Networks (ST GNNs) for traffic forecasting.

This repository investigates the effect of graph construction strategy on traffic forecasting performance while holding the backbone architecture and training procedure constant. The study evaluates three graph construction strategies under identical experimental settings using standard traffic forecasting benchmarks.

## Research Objective

The central objective is to isolate the effect of graph construction strategy in ST GNNs by comparing:

1. Predefined Static Graph
2. Learned Static Graph
3. Dynamic Graph

Evaluation is conducted on:

- METR LA
- PEMS BAY

under both:

- Standard forecasting settings
- Distributional shift settings

## Repository Structure

```
STGNN-Implementations/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── GraphWaveNet/
│   ├── README.md
│   └── ...
│
├── GraphWavenet(Static)/
│   ├── To_be_Implemented.md
│
└── GraphWavenet(DGCRN)/
    ├── To_be_Implemented.md
```