## Introduction

This repository contains the source code for Stateful Graph Diffusion Project. It is a small extenstion to publications [GRAND: Graph Neural Diffusion](https://icml.cc/virtual/2021/poster/8889).
The GRAND model treats deep learning on graphs as a continuous diffusion process and Graph Neural
Networks (GNNs) as discretisations of an underlying PDE. My work aims to disentangle layers weights in the GRAND model. 
In the original model, the weights are shared across layers and thus stateless. I use piecewise constant basis functions
to parameterize layer weights in GRAND and thus achieve the decoupling. 

## Running the experiments

### Requirements
Dependencies (with python >= 3.7):
Main dependencies are
torch>2.0
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.6
torch-sparse==0.6.9
torch-spline-conv==1.2.1
torchdiffeq==0.2.1
Commands to install all the dependencies in a new conda environment
```
conda create --name grand python=3.7
conda activate grand

pip install ogb pykeops
pip install torch==1.8.1
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric
```

### Dataset and Preprocessing
create a root level folder
```
./data
```
This will be automatically populated the first time each experiment is run.

### Experiments
For example to run for Cora with random splits:
```
cd src
python run_GNN_batch.py --dataset Cora --no_early --time 32 --block attention --method rk4 --epoch 100 --heads 1
```

## Code source
```
@article
{chamberlain2021grand,
  title={GRAND: Graph Neural Diffusion},
  author={Chamberlain, Benjamin Paul and Rowbottom, James and Goronova, Maria and Webb, Stefan and Rossi, 
  Emanuele and Bronstein, Michael M},
  journal={Proceedings of the 38th International Conference on Machine Learning,
               (ICML) 2021, 18-24 July 2021, Virtual Event},
  year={2021}
}
```
