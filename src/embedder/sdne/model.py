from src.core.embedder_base import Embedder
from src.embedder.sdne.sdne_model import SDNEModel
from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from torch.utils.data.dataloader import DataLoader
from numpy import ndarray
from networkx import to_numpy_array
from model import SDNEModel
import torch
import torch.optim as optim


class SDNEEmbedder(Embedder):

    def init(self):
        # todo move this to the configuration
        self.nhid0 = None
        self.nhid1 = None
        self.dropout = None
        self.alpha = None
        self.lr = None
        self.step_size = None
        self.gamma = None
        self.bs = None
        self.epochs = None
        self.beta = None
        self.nu1 = None
        self.nu2 = None

    def real_fit(self):
        pass

    def get_embeddings(self):
        result = [ self.get_embedding(x) for x in self.dataset.instances]
    
    def get_embedding(self, instance):
        return self._get_embedding(instance)

    def _get_embedding(self, instance: GraphInstance):
        n_nodes = instance.num_nodes
        adj_matrix = instance.get_nx().to_numpy_array()
        Adj = torch.FloatTensor(adj_matrix)
        #G, Adj, Node =  dataset.Read_graph(args.input)
        model = SDNEModel(n_nodes, self.nhid0, self.nhid1, self.dropout, self.alpha)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.gamma)
        torch_data = TorchGeometricDataset.to_geometric(instance)
        Data = DataLoader(torch_data, batch_size=self.bs, shuffle=True, )
        # todo update this part according to the project setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        for epoch in range(1, self.epochs + 1):
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            for index in Data:
                adj_batch = Adj[index]
                adj_mat = adj_batch[:, index]
                b_mat = torch.ones_like(adj_batch)
                b_mat[adj_batch != 0] = self.beta

                opt.zero_grad()
                L_1st, L_2nd, L_all = model(adj_batch, adj_mat, b_mat)
                L_reg = 0
                for param in model.parameters():
                    L_reg += self.nu1 * torch.sum(torch.abs(param)) + self.nu2 * torch.sum(param * param)
                Loss = L_all + L_reg
                Loss.backward()
                opt.step()
                loss_sum += Loss
                loss_L1 += L_1st
                loss_L2 += L_2nd
                loss_reg += L_reg
            scheduler.step(epoch)
            # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
            print("loss for epoch %d is:" %epoch)
            print("loss_sum is %f" %loss_sum)
            print("loss_L1 is %f" %loss_L1)
            print("loss_L2 is %f" %loss_L2)
            print("loss_reg is %f" %loss_reg)
        model.eval()
        embedding = model.savector(Adj)
        return embedding.detach().numpy()