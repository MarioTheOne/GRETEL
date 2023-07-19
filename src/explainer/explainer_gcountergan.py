import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, GCNConv

import wandb

from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class GraphCounteRGANExplainer(Explainer):
  
  def __init__(self,
               id,
               explainer_store_path,
               converter,
               n_nodes,
               n_labels=2,
               batch_size_ratio=0.1,
               training_iterations=20000,
               n_features=4,
               fold_id=0,
               sampling_iterations=10,
               lr_generator=0.001,
               lr_discriminator=0.001,
               config_dict=None) -> None:
    
    super().__init__(id, config_dict)
    
  
    self.name = 'graph_countergan'
    
    self.batch_size_ratio = batch_size_ratio
    self.n_labels = n_labels
    self.n_nodes = n_nodes
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.training_iterations = training_iterations
    
    self.explainer_store_path = explainer_store_path
    self.fold_id = fold_id
    self.converter = converter
    self.n_features = n_features
    self.sampling_iterations = sampling_iterations
    
    self.lr_discriminator = lr_discriminator
    self.lr_generator = lr_generator
    
    
    self._fitted = False

    # multi-class support
    self.explainers = [
        GraphCounteRGAN(n_nodes,
                        n_features=n_features,
                        residuals=True).to(self.device) for _ in range(n_labels)
    ]
    
  def _get_softmax_label(self, desired_label=0):
    y = -1
    while True:
      y = np.random.randint(low=0, high=self.n_labels)
      if y != desired_label:
        break
    return y
 

  def explain(self, instance, oracle: Oracle, dataset: Dataset):
    dataset = self.converter.convert(dataset)

    if(not self._fitted):
      self.fit(oracle, dataset, self.fold_id)

    # Getting the scores/class of the instance
    pred_scores = oracle.predict_proba(instance)

    with torch.no_grad():
      instance = dataset.get_instance(instance.id)
      adj = torch.from_numpy(instance.to_numpy_array())
      edge_list = torch.nonzero(torch.triu(adj))
      edge_attr = torch.tensor(instance.weights[edge_list[:, 0], edge_list[:, 1]])
      node_features = torch.tensor(instance.features)
      explainer = self.explainers[np.argmax(pred_scores)].generator
      explainer.eval()
      
      embedded_features, edge_probs = explainer(node_features, edge_list.T, edge_attr)
      ## Sample original instance edges
      # Get the number of samples to draw
      edge_num = instance.graph.number_of_edges()
      cf_instance = self.__sample(instance, embedded_features, edge_probs, edge_list, num_samples=edge_num) 
      if oracle.predict(cf_instance) != oracle.predict(instance):
        return cf_instance
      else:
        # get the "negative" edges that aren't estimated
        missing_edges = self.__negative_edges(edge_list, instance.graph.number_of_nodes())
        edge_probs = torch.from_numpy(np.array([1 / len(missing_edges) for _ in range(len(missing_edges))]))
        # check sampling for sampling_iterations
        # and see if we find a valid counterfactual
        for _ in range(self.sampling_iterations):
          cf_instance = self.__sample(cf_instance, embedded_features, edge_probs, missing_edges)
          if oracle.predict(cf_instance) != oracle.predict(instance):
            return cf_instance
      
      return instance

  def __negative_edges(self, edges, num_vertices):
    i, j = np.triu_indices(num_vertices, k=1)
    all_edges = set(list(zip(i, j)))
    edges = set([tuple(x) for x in edges])
    return torch.from_numpy(np.array(list(all_edges.difference(edges))))
  
  def __sample(self, instance: DataInstance, features, probabilities, edge_list, num_samples=1):
    adj = torch.zeros((self.n_nodes, self.n_nodes)).double()
    weights = torch.zeros((self.n_nodes, self.n_nodes)).double()
    ##################################################
    cf_instance = DataInstanceWFeaturesAndWeights(id=instance.id)
    try:    
      selected_indices = set(torch.multinomial(probabilities, num_samples=num_samples, replacement=True).numpy().tolist())
      selected_indices = np.array(list(selected_indices))
      
      adj[edge_list[selected_indices]] = 1
      adj = adj + adj.T - torch.diag(torch.diag(adj))
      
      weights[edge_list[selected_indices]] = probabilities[selected_indices]
      weights = weights + weights.T - torch.diag(torch.diag(weights))
      
      cf_instance.from_numpy_array(adj.numpy())
      cf_instance.weights = weights.numpy()
      cf_instance.features = features.numpy()
    except RuntimeError: # the probabilities are all zero
      cf_instance.from_numpy_array(adj.numpy())
    
    return cf_instance

  def save_explainers(self):
    for i, explainer in enumerate(self.explainers):
      torch.save(explainer.state_dict(),
                 os.path.join(self.explainer_store_path, self.name, f'explainer_{i}'))

  def load_explainers(self):
    for i in range(self.n_labels):
      self.explainers[i].load_state_dict(
        torch.load(
          os.path.join(self.explainer_store_path, self.name, f'explainer_{i}')
          )
      )


  def fit(self, oracle: Oracle, dataset : Dataset, fold_id=0):
    explainer_name = f'graph_countergan_fit_on_{dataset.name}_fold_id_{fold_id}'\
      + f'_lr_gen_{self.lr_generator}_lr_discr_{self.lr_discriminator}_epochs_{self.training_iterations}'\
        +f'_sampl_iters_{self.sampling_iterations}'
        
    explainer_uri = os.path.join(self.explainer_store_path, explainer_name)

    if os.path.exists(explainer_uri):
      # Load the weights of the trained model
      self.name = explainer_name
      self.load_explainers()

    else:
      # Create the folder to store the oracle if it does not exist
      os.mkdir(explainer_uri)        
      self.name = explainer_name
      
      for i in range(self.n_labels):
        self.__fit(self.explainers[i],
                  oracle,
                  dataset,
                  desired_label=i)

      self.save_explainers()        

    # setting the flag to signal the explainer was already trained
    self._fitted = True

  def _check_divergence(self, generated_features: torch.Tensor, generated_edge_probs: torch.Tensor):
      return torch.all(torch.isnan(generated_features)) or torch.all(torch.isnan(generated_edge_probs))
  
    
  def __fit(self, countergan, oracle : Oracle, dataset : Dataset, desired_label=0):
    generator_loader, discriminator_loader = self.transform_data(dataset, oracle, class_to_explain=desired_label)
  
    discriminator_optimizer = torch.optim.Adam(countergan.discriminator.parameters(), lr=self.lr_discriminator)
    generator_optimizer = torch.optim.Adam(countergan.generator.parameters(), lr=self.lr_generator)
    
    loss = nn.BCELoss()
    
    for iteration in range(self.training_iterations):
      G_losses, D_losses = [], []

      if iteration > 0:
        encoded_features, edge_probs = countergan.generator(features, edge_index, edge_attr)
        if self._check_divergence(encoded_features, edge_probs):
          break
      
      
      for features, edge_index, edge_attr, real_label in discriminator_loader:
        countergan.set_training_generator(False)
        countergan.set_training_discriminator(True)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        countergan.discriminator.zero_grad()
        # Format batch
        features = features[1].squeeze(dim=0).to(self.device)
        edge_index = edge_index[1].squeeze(dim=0).to(self.device)
        edge_attr = edge_attr[1].squeeze(dim=0).to(self.device)
        real_label = real_label[1].to(self.device)
        
        label = torch.full((self.batch_size,), real_label.item(), dtype=torch.float64, device=self.device)

        # Forward pass real batch through D
        output = countergan.discriminator(features, edge_index, edge_attr).view(-1)
        # Calculate loss on all-real batch
        errD_real = loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        
        ## Train with all fake batch
        for fake_features, fake_edge_index, fake_edge_attr, fake_label in generator_loader:
          countergan.set_training_discriminator(True)
          countergan.set_training_generator(False)
          # Format batch of fake graph data
          fake_features = fake_features[1].squeeze(dim=0).to(self.device)
          fake_edge_index = fake_edge_index[1].squeeze(dim=0).to(self.device)
          fake_edge_attr = fake_edge_attr[1].squeeze(dim=0).to(self.device)
          fake_label = fake_label[1].to(self.device)
          
          label.fill_(fake_label.item())
          
          # put the fake data through the generator
          embed_features, edge_probs = countergan.generator(fake_features, fake_edge_index, fake_edge_attr)
          # Classify fake batch with D
          output = countergan.discriminator(embed_features.detach(), fake_edge_index.detach(), edge_probs.detach()).view(-1)
          # Calculate D's loss on all-fake graphs
          errD_fake = loss(output, label)
          # Calculate the gradients for this batch, accumulated with previous grads
          errD_fake.backward()
          D_G_z1 = output.mean().item()
          # Compute error of D as sum over fake and real batches
          errD = errD_real + errD_fake
          # Update D
          discriminator_optimizer.step()
          
          ###################################
          ## Update G network: maximize log(D(G(z)))
          ###################################
          countergan.set_training_generator(True)
          countergan.set_training_discriminator(False)
                              
          countergan.generator.zero_grad()
          # fake labels are real for generator cost
          label.fill_(real_label.item())
          # Since we just updated D, perform another forward pass of all-fake batch through D
          output = countergan.discriminator(embed_features, fake_edge_index, edge_probs).view(-1)
          # Calculate G's loss based on this output
          errG = loss(output, label)
          # Calculate gradients for G
          errG.backward()
          D_G_z2 = output.mean().item()
          # Update G
          generator_optimizer.step()

          # Save Losses for plotting later
          G_losses.append(errG.item())
          D_losses.append(errD.item())
          
        print(f'Iteration {iteration}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
          
        wandb.log({
          f'iteration_cls={desired_label}': iteration,
          f'loss_d_cls={desired_label}_{self.fold_id}': np.mean(D_losses),
          f'loss_g_cls={desired_label}_{self.fold_id}': np.mean(G_losses)
        })
      
      
  def transform_data(self, dataset: Dataset, oracle: Oracle, class_to_explain=0):
    adj  = torch.from_numpy(np.array([i.to_numpy_array() for i in dataset.instances]))
    features = torch.from_numpy(np.array([i.features for i in dataset.instances]))
    weights = torch.from_numpy(np.array([i.weights for i in dataset.instances]))
    y = torch.from_numpy(np.array([oracle.predict(i) for i in dataset.instances]))
        
    indices = dataset.get_split_indices()[self.fold_id]['train'] 
    adj, features, weights, y = adj[indices], features[indices], weights[indices], y[indices]
    
    # weights need to be positive
    weights = torch.abs(weights)
 
    data_list = []
    w = None
    a = None
    for i in range(len(y)):
      # weights is an adjacency matrix n x n x d
      # where d is the dimensionality of the edge weight vector
      # get all non zero vectors. now the shape will be m x d
      # where m is the number of edges and 
      # d is the dimensionality of the edge weight vector
      w = weights[i]
            
      # get the edge indices
      # shape m x 2
      a = torch.nonzero(torch.triu(adj[i]))
      w = w[a[:,0], a[:,1]]
            
      data_list.append(Data(x=features[i], y=y[i], edge_index=a.T, edge_attr=w))

    class_to_explain_indices = (y == class_to_explain).nonzero(as_tuple=True)[0].numpy()
    class_to_not_explain_indices = (y != class_to_explain).nonzero(as_tuple=True)[0].numpy()
    data_list = np.array(data_list, dtype=object)
        
    generator_data = GraphCounteRGANDataset(data_list[class_to_explain_indices].tolist())
    discriminator_data = GraphCounteRGANDataset(data_list[class_to_not_explain_indices].tolist())    
    
    #### change in the future to handle more than 1 graph
    self.batch_size = 1
    
    generator_loader = DataLoader(generator_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2)
    
    discriminator_loader = DataLoader(discriminator_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
  
    return generator_loader, discriminator_loader
        
class GraphCounteRGAN(nn.Module):
  
  def __init__(self, n_nodes=28,
               residuals=True,
               n_features=1):
    super(GraphCounteRGAN, self).__init__()
        
    self.n_nodes = n_nodes
    self.residuals = residuals
    
    self.generator = ResidualGenerator(n_features=n_features,
                                       residuals=residuals)
    self.generator.double()
    
    self.discriminator = Discriminator(n_nodes=n_nodes,
                                       n_features=n_features)
    self.discriminator.double()
        
  def set_training_discriminator(self, training):
    self.discriminator.set_training(training)
      
  def set_training_generator(self, training):
    self.generator.set_training(training)
      
  def forward(self, features, edge_index, edge_attr):
    features, edge_probs = self.generator(features, edge_index, edge_attr)
    return self.discriminator(features, edge_index, edge_probs)


class Discriminator(nn.Module):

  def __init__(self, n_nodes=28, n_features=1):
    """This class provides a GCN to discriminate between real and generated graph instances"""
    super(Discriminator, self).__init__()

    self.training = False
    self.n_nodes = n_nodes

    self.conv1 = GCNConv(n_features, 64)
    #self.conv2 = GCNConv(64, 64)
    #self.conv3 = GCNConv(64, 64)
    self.fc = nn.Linear(self.n_nodes * 64, 1)
    
    self.device = "cuda" if torch.cuda.is_available() else "cpu"


  def set_training(self, training):
    self.training = training

  def forward(self, x, edge_list, edge_attr):
    x = x.double()
    edge_attr = edge_attr.double()
    x = self.conv1(x, edge_list, edge_attr)

    if self.training:
      x = self.add_gaussian_noise(x)

    x = F.relu(x)
    x = F.dropout(x, p=.4, training=self.training)
    """x = self.conv2(x, edge_list, edge_attr)
    x = F.relu(x)
    x = F.dropout(x, p=.4, training=self.training)
    x = self.conv3(x, edge_list, edge_attr)"""
    #x = F.relu(x)
    #x = F.dropout(x, p=.4, training=self.training).view(1, self.n_nodes, -1)
    x = torch.flatten(x)
    x = self.fc(x)
    x = torch.sigmoid(x).squeeze()

    return x

  def add_gaussian_noise(self, x, sttdev=0.2):
    noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
    return x + noise
  

class ResidualGenerator(nn.Module):

  def __init__(self, n_features, residuals=True):
    super(ResidualGenerator, self).__init__()

    self.n_features = n_features
    self.gcn_encoder = GCNGeneratorEncoder(self.n_features, 64)
    self.gcn_encoder.double()
    self.model = GAE(encoder=self.gcn_encoder)
    self.residuals = residuals

  def set_training(self, training):
    self.gcn_encoder.set_training(training)
    
  def forward(self, node_features, edge_list, edge_attr):
    encoded_node_features = self.model.encode(node_features, edge_list, edge_attr)
    edge_probabilities = self.model.decode(encoded_node_features, edge_list)
    
    if self.residuals:
      encoded_node_features = torch.add(encoded_node_features, node_features)
      
    edge_probabilities = torch.nan_to_num(edge_probabilities, 0)

    return encoded_node_features, edge_probabilities
  
class GCNGeneratorEncoder(nn.Module):
  """This class use GCN to generate an embedding of the graph to be used by the generator"""

  def __init__(self, in_channels=1, out_channels=64):
    super().__init__()
    self.conv1 = GCNConv(in_channels, out_channels)
    self.conv2 = GCNConv(out_channels, in_channels)
    #self.conv3 = GCNConv(out_channels, in_channels)
    
    self.training = False

  def forward(self, x, edge_index, edge_attr):
    x = x.double()
    edge_attr = edge_attr.double()
    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)
    x = F.dropout(x, p=.2, training=self.training)
    x = self.conv2(x, edge_index, edge_attr)
    x = torch.sigmoid(x)
    #x = F.dropout(x, p=.2, training=self.training)
    #x = self.conv3(x, edge_index, edge_attr)
    #x = torch.tanh(x)
    return x

  def set_training(self, training):
    self.training = training

class GraphCounteRGANDataset(GeometricDataset):
  
  def __init__(self, instances):
    super(GeometricDataset, self).__init__()
    self.instances = instances
    
  def __len__(self):
    return len(self.instances)
  
  def __getitem__(self, idx):
    return self.instances[idx]
