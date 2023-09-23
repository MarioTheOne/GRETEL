#!/bin/bash
conda update -n base -c defaults conda
conda create -n $1-CUDA python=3.9 -y 

pip install -y tensorflow tensorrt

pip install -y torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -y exmol gensim joblib jsonpickle karateclub matplotlib networkx numpy pandas rdkit scikit-learn scipy selfies sqlalchemy black typing-extensions torch_geometric dgl IPython ipykernel flufl.lock
