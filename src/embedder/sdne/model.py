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
        self.nhid0 = self.local_config['parameters']['nhid0']
        self.nhid1 = self.local_config['parameters']['nhid1']
        self.dropout = self.local_config['parameters']['dropout']
        self.alpha = self.local_config['parameters']['alpha']
        self.lr = self.local_config['parameters']['lr']
        self.step_size = self.local_config['parameters']['step_size']
        self.gamma = self.local_config['parameters']['gamma']
        self.bs = self.local_config['parameters']['bs']
        self.epochs = self.local_config['parameters']['epochs']
        self.beta = self.local_config['parameters']['beta']
        self.nu1 = self.local_config['parameters']['nu1']
        self.nu2 = self.local_config['parameters']['nu2']

    def real_fit(self):
        self.model = { instance.id:self._train_embedding(instance) for instance in self.dataset.instances }

    def get_embeddings(self):
        return self.model.values()
    
    def get_embedding(self, instance):
        return self.model[instance.id]

    def _train_embedding(self, instance: GraphInstance):
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
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['nhid0'] =  self.local_config['parameters'].get('nhid0', 1000)
        self.local_config['parameters']['nhid1'] =  self.local_config['parameters'].get('nhid1', 128)
        self.local_config['parameters']['dropout'] =  self.local_config['parameters'].get('dropout', 0.5)
        self.local_config['parameters']['alpha'] =  self.local_config['parameters'].get('alpha', 1e-2)
        self.local_config['parameters']['gamma'] =  self.local_config['parameters'].get('gamma', 0.9)
        self.local_config['parameters']['lr'] =  self.local_config['parameters'].get('lr', 0.001)
        self.local_config['parameters']['step_size'] =  self.local_config['parameters'].get('step_size', 10)
        self.local_config['parameters']['bs'] =  self.local_config['parameters'].get('bs', 100)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 100)
        self.local_config['parameters']['beta'] =  self.local_config['parameters'].get('beta', 5.)
        self.local_config['parameters']['nu1'] =  self.local_config['parameters'].get('nu1', 1e-5)
        self.local_config['parameters']['nu2'] =  self.local_config['parameters'].get('nu2', 1e-4)