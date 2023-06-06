import os
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.nn import DenseGCNConv, DenseGraphConv

from src.dataset.converters.causality_converter import \
    DefaultCausalityConverter
from src.dataset.data_instance_features import DataInstanceWFeatures
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class CLEARExplainer(Explainer):
    
    def __init__(self,
                 id,
                 explainer_store_path,
                 n_nodes,
                 n_labels=2,
                 batch_size_ratio=.1,
                 h_dim=10,
                 z_dim=10,
                 dropout=.1,
                 encoder_type='gcn',
                 graph_pool_type='mean',
                 disable_u=False,
                 epochs=200,
                 alpha=5,
                 feature_dim=2,
                 lr=1e-3,
                 weight_decay=1e-5,
                 lambda_sim=1,
                 lambda_kl=1,
                 lambda_cfe=1,
                 beta_x=10,
                 beta_adj=10,
                 fold_id=0,
                 config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.name = 'clear'
        
        self.n_labels = n_labels
        self.n_nodes = n_nodes
        self.batch_size_ratio = batch_size_ratio
        self.explainer_store_path = explainer_store_path
        self.fold_id = fold_id
        self.lambda_sim = lambda_sim
        self.lambda_kl = lambda_kl
        self.lambda_cfe = lambda_cfe
        self.epochs = epochs
        self.alpha = alpha
        self.beta_x = beta_x
        self.beta_adj = beta_adj
        self.feature_dim = feature_dim
                
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.explainer = CLEAR(feature_dim=feature_dim,
                               n_nodes=n_nodes,
                               graph_pool_type=graph_pool_type,
                               encoder_type=encoder_type,
                               h_dim=h_dim,
                               z_dim=z_dim,
                               dropout=dropout,
                               disable_u=disable_u,
                               device=self.device
                            ).to(self.device)
                        
        self.optimizer = torch.optim.Adam(self.explainer.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        
        self.converter = DefaultCausalityConverter(causality_dim_choice=10)   

        self._fitted = False
                
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        dataset = self.converter.convert(dataset)      
        
        if(not self._fitted):
            self.fit(oracle, dataset, self.fold_id)

        instance = dataset.get_instance(instance.id)
        self.explainer.eval()
        
        with torch.no_grad():
            features = torch.from_numpy(np.array(instance.features)).float().to(self.device)
            adj = torch.from_numpy(instance.to_numpy_array()).float().to(self.device)
            u = torch.from_numpy(np.array(instance.causality)).float().to(self.device)[None,:]
            labels = torch.from_numpy(np.array([instance.graph_label])).to(self.device)[None,:]
            
            model_return = self.explainer(features, u, adj, labels)
            adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']
            
            adj_reconst_binary = torch.bernoulli(adj_reconst.squeeze())
            
            cf_instance = DataInstanceWFeatures(instance.id)
            cf_instance.from_numpy_array(adj_reconst_binary.detach().numpy())
            cf_instance.features = features_reconst.squeeze().detach().numpy()
            
            print(f'Finished evaluating for instance {instance.id}')
            return cf_instance

    def save_explainers(self):
        torch.save(self.explainer.state_dict(),
                   os.path.join(self.explainer_store_path, self.name, f'explainer'))
 
    def load_explainers(self):
        self.explainer.load_state_dict(torch.load(
            os.path.join(self.explainer_store_path, self.name, f'explainer')))

    def fit(self, oracle: Oracle, dataset : Dataset, fold_id=0):
        explainer_name = 'clear_fit_on_' + dataset.name + '_fold_id_' + str(fold_id)
        explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name
        
        if os.path.exists(explainer_uri):
            # Load the weights of the trained model
            self.load_explainers()
        else:
            # Create the folder to store the oracle if it does not exist
            os.mkdir(explainer_uri)                    
            self.__fit(oracle, dataset, fold_id)
            self.save_explainers()        
        # setting the flag to signal the explainer was already trained
        self._fitted = True

    def __fit(self, oracle, dataset, fold_id):
        train_loader = self.transform_data(dataset, fold_id=fold_id)
        
        for epoch in range(self.epochs):
            self.explainer.train()
            
            batch_num = 0
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0, 0, 0, 0, 0
            for _, data in enumerate(train_loader):
                batch_num += 1
                                
                adj, features, u, labels = data
                features = features.float().to(self.device)
                u = u.float().to(self.device)
                adj = adj.float().to(self.device)
                labels = (1 - labels.float()).to(self.device)
            
                self.optimizer.zero_grad()
                # forward pass
                retr = self.explainer(features, u, adj, labels)
                # z_cf
                z_mu_cf, z_logvar_cf = self.explainer.get_represent(
                    retr['features_reconst'], 
                    u, 
                    retr['adj_reconst'], 
                    labels)
                # compute loss
                loss_params = {
                    'model': self.explainer,
                    'oracle': oracle,
                    'adj_input': adj,
                    'features_input': features,
                    'y_cf': labels,
                    'z_mu_cf': z_mu_cf,
                    'z_logvar_cf': z_logvar_cf
                }
                loss_params.update(retr)
                
                loss_results = self.__compute_loss(loss_params)
                loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'],\
                    loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
                    
                loss += loss_batch
                loss_kl += loss_kl_batch
                loss_sim += loss_sim_batch
                loss_cfe += loss_cfe_batch
                loss_kl_cf += loss_kl_batch_cf
                
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num, loss_kl_cf / batch_num
            print(f'Epoch {epoch+1} ---> loss {loss}')
            # backward
            alpha = self.alpha if epoch >= 450 else 0
            ((loss_sim + loss_kl + alpha * loss_cfe) / batch_num).backward()        
            self.optimizer.step()
        
        
    def __compute_loss(self, params):
        model, oracle, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
            adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['model'], params['oracle'], params['z_mu'], \
                params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
                    params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']
                    
        # kl loss
        loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
        loss_kl = torch.mean(loss_kl)
        
        # similarity loss
        size = len(features_permuted)
        dist_x = torch.mean(self.__distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
        dist_a = self.__distance_graph_prob(adj_permuted, adj_reconst)
                
        loss_sim = self.beta_x * dist_x + self.beta_adj * dist_a
        
        # CFE loss
        temp_instance = DataInstanceWFeatures(-1)
        y_pred = []
        for i in range(len(adj_reconst)):
            temp_instance.from_numpy_array(
                adj_reconst[i].detach().numpy().squeeze()
            )
            temp_instance.features = features_reconst[i].detach().numpy().squeeze()
            y_pred.append(oracle.predict_proba(temp_instance))
            
        y_pred = torch.from_numpy(np.array(y_pred)).float().squeeze()
        loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())
        
        # rep loss
        if z_mu_cf is None:
            loss_kl_cf = 0
        else:
            loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
            loss_kl_cf = torch.mean(loss_kl_cf)
            
        loss = self.lambda_sim * loss_sim + self.lambda_kl * loss_kl + self.lambda_cfe * loss_cfe

        loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
        return loss_results  
    
    
    def __distance_feature(self, feat_1, feat_2):
        pdist = nn.PairwiseDistance(p=2)
        return pdist(feat_1, feat_2) / 4
    
    def __distance_graph_prob(self, adj_1, adj_2_prob):
        return F.binary_cross_entropy(adj_2_prob, adj_1)
    
    def transform_data(self, dataset: Dataset, fold_id=0):             
        X_adj  = np.array([i.to_numpy_array() for i in dataset.instances])
        X_features = np.array([i.features for i in dataset.instances])
        X_causality = np.array([i.causality for i in dataset.instances])
        y = np.array([i.graph_label for i in dataset.instances])[..., np.newaxis]
        
        X_adj = X_adj[dataset.get_split_indices()[fold_id]['train']]
        X_features = X_features[dataset.get_split_indices()[fold_id]['train']]
        X_causality = X_causality[dataset.get_split_indices()[fold_id]['train']]
        y_train = y[dataset.get_split_indices()[fold_id]['train']]
        
        dataset = TensorDataset(
            torch.tensor(X_adj, dtype=torch.float),
            torch.tensor(X_features, dtype=torch.float),
            torch.tensor(X_causality, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float)
            )
        
        loader = DataLoader(dataset,
                            batch_size=int(len(X_adj) * self.batch_size_ratio),
                            shuffle=True,
                            num_workers=2,
                            drop_last=True)
        
        return loader
    
        
class CLEAR(nn.Module):

    def __init__(self,
                 feature_dim,
                 h_dim=16,
                 z_dim=16,
                 dropout=False,
                 n_nodes=10,
                 encoder_type='gcn',
                 graph_pool_type='mean',
                 disable_u=False,
                 device='cuda'
                ):
        super(CLEAR, self).__init__()
        
        self.x_dim = feature_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.encoder_type = encoder_type
        self.graph_pool_type = graph_pool_type
        self.disable_u = disable_u
        self.device = device
        self.u_dim = 1 # init_params['u_dim']
        
        if self.disable_u:
            self.u_dim = 0
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        else:
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)
        
        
        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim,
                              n_layers=1, activation='none', slope=.1,
                              device=self.device)
        
        self.prior_var = nn.Sequential(
            MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1,
                activation='none', slope=.1, device=self.device),
            nn.Sigmoid()
        )
        
        # encoder
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU()
        )
        
        self.encoder_var = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU(),
            nn.Sigmoid()
        )
        
        # decoder
        self.decoder_x = nn.Sequential(
            nn.Linear(self.z_dim + 1, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_nodes * self.x_dim)
        )
        
        in_channels_a = self.z_dim + 1 if self.disable_u else self.z_dim + 2
        self.decoder_a = nn.Sequential(
            nn.Linear(in_channels_a, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_nodes * self.n_nodes),
            nn.Sigmoid()
        )
        
        self.grpah_norm = nn.BatchNorm1d(self.h_dim)
        
        
    def encoder(self, features, u, adj, y_cf):
        # Q(Z | X, U, A, Y^CF)
        # input: x, u, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj) # n x num_node x h_dim
        graph_rep  = self.graph_pooling(graph_rep, self.graph_pool_type) # n x h_dim
        # graph_rep = self.graph_norm(graph_rep)
        
        if self.disable_u:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))
            
        return z_mu, z_logvar
    
    def get_represent(self, features, u, adj, y_cf):
        u_onehot = u
        return self.encoder(features, u_onehot, adj, y_cf)
    
    def decoder(self, z, y_cf, u):
        if self.disable_u:
            adj_reconst = self.decoder_a(
                torch.cat((z, y_cf), dim=1)
                ).view(-1, self.n_nodes, self.n_nodes)
        else:
            adj_reconst = self.decoder_a(
                torch.cat((z, u, y_cf), dim=1)
                ).view(-1, self.n_nodes, self.n_nodes)
                            
        features_reconst = self.decoder_x(
            torch.cat((z, y_cf), dim=1)
            ).view(-1, self.n_nodes, self.x_dim)
        
        return features_reconst, adj_reconst
    
    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out
    
    def prior_params(self, u): # P(Z | U)
        if self.disable_u:
            z_u_mu = torch.zeros((len(u), self.h_dim)).to(self.device)
            z_u_logvar = torch.ones((len(u), self.h_dim)).to(self.device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
            
        return z_u_mu, z_u_logvar
    
    
    def reparametrize(self, mu, logvar):
        # compute z = mu + std * epsilon
        if self.training:
            # compute std from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with
            # mean 0 and variance 1
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
        
    def forward(self, features, u, adj, y_cf):
        u_onehot = u
        
        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparametrize
        z_sample = self.reparametrize(z_mu, z_logvar)
        # decoder
        
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)
        
        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'adj_permuted': adj,
            'features_permuted': features,
            'adj_reconst': adj_reconst,
            'features_reconst': features_reconst,
            'z_u_mu': z_u_mu,
            'z_u_logvar': z_u_logvar
        }
        
        

class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers,
                 activation='none', slope=.1, device='cuda'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.slope = slope
        self.device = device
        
        
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(f'Wrong argument type for hidden_dim {hidden_dim}')
        
        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError(f'Wrong argument type for activation {activation}')
        
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError(f'Incorrect activation: {act}')

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)
        
        
    @staticmethod
    def xtanh(x, alpha=.1):
        return x.tanh() + alpha * x
    
    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h