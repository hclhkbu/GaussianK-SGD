# GaussianK-SGD
## Introduction
This repository contains the codes for the paper: Understanding Top-k Sparsification in Distributed Deep Learning. Key features include
- Distributed training with gradient sparsification.
- Measurement of gradient distribution on various deep learning models including feed foward neural networks (FFNs), CNNs and LSTMs.
- A computing-efficient top-k approximation (called gaussian-k) for gradient sparsification.

For more details about the algorithm, please refer to our papers.

## Installation
### Prerequisites
- Python 2 or 3
- PyTorch-0.4.+
- [OpenMPI-3.1.+](https://www.open-mpi.org/software/ompi/v3.1/)
- [Horovod-0.14.+](https://github.com/horovod/horovod)
### Quick Start
```
git clone https://github.com/hclhkbu/GaussianK-SGD.git
cd GaussianK-SGD
HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod (optional if horovod has been installed)
pip install -r requirements.txt
dnn=resnet20 nworkers=4 compressor=topk density=0.001 ./run.sh
```
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the ResNet-20 model with the Cifar-10 data set using SGD with top-k sparsification.

## Papers
- S. Shi, X.-W. Chu, K. Cheung and S. See, “Understanding Top-k Sparsification in Distributed Deep Learning,” 2019.

## Referred Models
- Deep speech: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
- PyTorch examples: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
