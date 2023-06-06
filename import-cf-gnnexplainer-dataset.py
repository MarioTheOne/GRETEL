import argparse
import pickle

import torch
from dgl.data.utils import load_graphs

from experimental.cfgnnexplainer.src.gcn import GCNSynthetic
from src.dataset.dataset_syn import NodeDataset, SynDataset
from oracle.oracle_node import NodeOracle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn5')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip).')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

for ds in ["1", "4", "5"]:
    with open("experimental/cfgnnexplainer/data/gnn_explainer/syn{}.pickle".format(ds), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])

    model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
                        nclass=len(labels.unique()), dropout=args.dropout)

    dataset = SynDataset("syn" + ds)
    dataset.name = "syn" + ds

    for i in idx_test:
        dataset.instances.append(NodeDataset(name = i.item(), graph_data=data, target_node=i.item()))

    dataset.write_data("/home/coder/gretel/data/datasets")

    d = SynDataset(args.dataset)
    d.read_data(f"/home/coder/gretel/data/datasets/syn{ds}")
    print(d.name)



oracle = PtGCNSyntheticOracle(id="gcn_syn_pt",oracle_store_path = "/home/coder/gretel/data/oracles")
oracle.fit(d)


