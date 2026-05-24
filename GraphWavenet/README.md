
## GraphWaveNet (GWNet) Results

This folder contains results for a trained GWNet model and its incremental improvement. 

### Implementation Notes
The Graph WaveNet implementation is based on the publicly available Graph WaveNet codebase. The original training framework and backbone architecture were retained; however, minor implementation adjustments were made to ensure compatibility and correct execution. In particular, the graph convolution operation was updated from `nn.conv1d` to `nn.conv2d` to resolve tensor dimensional misalignment issues encountered during implementation. No changes were made to the loss function or optimisation procedure. The experimental setup and evaluation pipeline were adapted for this implementation.

### Folder structure
```
GraphWaveNet/
│
├── README.md
├── engine.py
├── generate_training_data.py
├── model.py
├── train.py
├── util.py
├── LICENSE
├── .gitignore
│
└── Model_Output/
    ├── best_model.pth
    ├── test.csv
    └── metrics.csv
```
### Environment
- Python: 3.10
- PyTorch: 2.5.1 + CUDA 12.1
- Device: NVIDIA GeForce GTX 1060 3GB
- Virtual Environment: Python `venv`
- Environment configuration provided in `environment.yml`
- Package requirements provided in `requirements.txt`

### Results

#### Model: Graph WaveNet (Wu et al., 2019)

- Early stopping: Disabled
- Total epochs: 20
- Best epoch: 18
- Criterion: Validation loss
- Best validation loss: 2.8168
- Average training time: 129.84 s/epoch
- Average inference time: 5.64 s

| Split      | MAE      | MAPE    | RMSE    |
| ---------- | -------- | ------- | ------- |
| Train      | 2.9845   | 0.0811  | 5.9562  |
| Validation | 2.8168   | 0.0787  | 5.3875  |
| Test       | **3.1154** | **0.0847** | **6.1534** |



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

