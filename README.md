# Graph-Wavenet-Implementation
Empirical investigation of graph construction strategy in ST-GNNs for traffic forecasting. Modified Graph WaveNet evaluated under three conditions, static, adaptive, and dynamic, on METR-LA and PEMS-BAY under standard and distributional shift settings.

## GraphWaveNet (GWNet) Results

This folder contains results for a trained GWNet model and its incremental improvement. 

### Implementation notes
We used the publicly available GWNet () codebases as our training frameworks. No changes were made to model architectures, loss functions or optimisations. 

### Repo structure
```
GWNet-results/
├── README.md
├── Wu/
│   └── `best_model.pth`: PyTorch checkpoint of the trained model
│   └── `test.csv`: test set predictions and ground truth
│   └── `metrics.csv`: evaluation metrics on the test set
│   └── `args.pkl`: pickled Python object storing the full set of training arguments used to produce `best_model.pth`
├── Schleifer/
│   └── `best_model.pth`: PyTorch checkpoint of the trained model
│   └── `test.csv`: test set predictions and ground truth
│   └── `metrics.csv`: evaluation metrics on the test set
│   └── `args.pkl`: pickled Python object storing the full set of training arguments used to produce `best_model.pth`
└── requirements.txt
└── environment.yaml
```
### Environment
- Python:  3.9
- PyTorch: 1.10.2
- Device: CPU (Apple Silicon, 8GB Unified Memory)
- Conda environment exported in `environment.yaml`

### Results
#### Model: Graph WaveNet (Wu et al., 2019)
- Early stopping: enabled
- Best epoch: 64
- Training stopped at: 84
- Criterion: validation loss

| Split      | MAE       | MAPE      | RMSE      |
| ---------- | --------- | --------- | --------- |
| Train      |    T.BA  |   T.BA   |   T.BA   |
| Validation |    T.BA  |   T.BA  |   T.BA  |
| Test       | **T.BA** | **T.BA** | **T.BA** |

#### Model: Graph WaveNet (Schleifer et al., 2019)
- Early stopping: enabled
- Best epoch: 38
- Training stopped at: 58
- Criterion: validation loss

| Split      | MAE       | MAPE      | RMSE      |
| ---------- | --------- | --------- | --------- |
| Train      | 2.734     | 0.074     | 5.568     |
| Validation | 2.739     | 0.076     | 5.341     |
| Test       | **3.025** | **0.083** | **6.061** |



### References

The implementation for GWNet (Wu et. al) is based on the IJCAI 2019 paper “Graph WaveNet for Deep Spatial-Temporal Graph Modeling” (https://arxiv.org/abs/1906.00121), with the following citation:

```bibtex
@article{wu2019graph,
  title={Graph wavenet for deep spatial-temporal graph modeling},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  journal={arXiv preprint arXiv:1906.00121},
  year={2019}
}
```

The implementation for GWNet (Schleifer et al.) is based on the arXiv 2019 paper “Incrementally Improving Graph WaveNet Performance on Traffic Prediction” (https://arxiv.org/abs/1912.07390), with the following citation:

```bibtex
@article{shleifer2019incrementally,
  title={Incrementally improving graph wavenet performance on traffic prediction},
  author={Shleifer, Sam and McCreery, Clara and Chitters, Vamsi},
  journal={arXiv preprint arXiv:1912.07390},
  year={2019}
}
```
