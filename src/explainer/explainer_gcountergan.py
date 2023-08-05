import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, GCNConv

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.utils.samplers.abstract_sampler import Sampler
from src.utils.samplers.partial_order_samplers import \
    PositiveAndNegativeEdgeSampler
    
from torch import Tensor

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
               sampler: Sampler = PositiveAndNegativeEdgeSampler(10),
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
    
    self.sampler = sampler
    
    self._fitted = False

    # multi-class support
    self.explainers = [
        GraphCounteRGAN(n_nodes,
                        n_features=n_features,
                        residuals=True).to(self.device) for _ in range(n_labels)
    ]

  def explain(self, instance, oracle: Oracle, dataset: Dataset):
    dataset = self.converter.convert(dataset)

    if(not self._fitted):
      self.fit(oracle, dataset, self.fold_id)

    # Getting the scores/class of the instance
    pred_label = oracle.predict(instance)

    with torch.no_grad():
      instance = dataset.get_instance(instance.id)
      batch = TorchGeometricDataset.to_geometric(instance)
      explainer = self.explainers[pred_label].generator
      embedded_features, _, edge_probs = explainer(batch.x, batch.edge_index, batch.edge_attr)
      print(batch.x)
      cf_instance = self.sampler.sample(instance, oracle, **{'embedded_features': embedded_features,
                                                             'edge_probabilities': edge_probs,
                                                             'edge_index': batch.edge_index})
      
    return cf_instance if cf_instance else instance

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
        self.__fit(self.explainers[i], oracle, dataset, desired_label=i)

      self.save_explainers()        

    # setting the flag to signal the explainer was already trained
    self._fitted = True

  def _check_divergence(self, generated_features: torch.Tensor, generated_edge_probs: torch.Tensor):
      return torch.all(torch.isnan(generated_features)) or torch.all(torch.isnan(generated_edge_probs))
  
  def _infinite_data_stream(self, loader: DataLoader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch
    
  def __format_batch(self, batch):
    # Format batch
    features = batch.x.to(self.device)
    edge_index = batch.edge_index.squeeze(dim=0).to(self.device)
    edge_attr = batch.edge_attr.to(self.device)
    label = batch.y.to(self.device)
    
    return features, edge_index, edge_attr, label
  
  def __fit(self, countergan, oracle : Oracle, dataset : Dataset, desired_label=0):
    print(f'Training for desired label = {desired_label}')
    generator_loader, discriminator_loader = self.transform_data(dataset, oracle, class_to_explain=desired_label)
    generator_loader = self._infinite_data_stream(generator_loader)
    discriminator_loader = self._infinite_data_stream(discriminator_loader)
    
    discriminator_optimizer = torch.optim.Adam(countergan.discriminator.parameters(), lr=self.lr_discriminator)
        
    countergan_optimizer = torch.optim.NAdam(countergan.parameters(), lr=self.lr_generator)
    
    loss_discriminator = nn.BCELoss(reduction='none')
    loss_countergan = nn.BCELoss()

    for iteration in range(self.training_iterations):
      G_losses, D_losses = [], []

      if iteration > 0:
        encoded_features, _, edge_probs = countergan.generator(features, edge_index, edge_attr)
        if self._check_divergence(encoded_features, edge_probs):
          break
      #######################################################################
      discriminator_optimizer.zero_grad()
      countergan.set_training_generator(False)
      countergan.generator.train(False)
      countergan.set_training_discriminator(True)
      countergan.discriminator.train(True)
      #######################################################################
      # discriminator data (real batch)
      features, edge_index, edge_attr, _ = self.__format_batch(next(discriminator_loader))
      #######################################################################
      # generator data (fake batch)
      fake_features, fake_edge_index, fake_edge_attr, _ = self.__format_batch(next(generator_loader))
      _, fake_edge_index, edge_probs = countergan.generator(fake_features, fake_edge_index, fake_edge_attr)
      # get the real and fake labels
      y_batch = torch.cat([torch.ones((self.batch_size,)), torch.zeros((self.batch_size,))], dim=0)
      #######################################################################
      # get the oracle's predictions
      oracle_scores = []
      inst1, inst2 = DataInstance(-1), DataInstance(-1)
      inst1.from_numpy_array(rebuild_adj_matrix(len(features), edge_index, edge_attr).numpy())
      inst2.from_numpy_array(rebuild_adj_matrix(len(fake_features), fake_edge_index, fake_edge_attr).numpy())
      oracle_scores = [oracle.predict_proba(inst)[1-desired_label] for inst in [inst1, inst2]]
      # The following update to the oracle scores is needed to have
      # the same order of magnitude between real and generated sample losses
      oracle_scores = np.array(oracle_scores, dtype=float).squeeze()
      real_samples = torch.where(y_batch == 1.)
      average_score_real_samples = np.mean(oracle_scores[real_samples])
      if average_score_real_samples != 0:
          oracle_scores[real_samples] /= average_score_real_samples
      
      fake_samples = torch.where(y_batch == 0.)
      oracle_scores[fake_samples] = 1.
      
      #######################################################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      real_pred = countergan.discriminator(features, edge_index, edge_attr).expand(1)
      fake_pred = countergan.discriminator(fake_features, fake_edge_index, fake_edge_attr).expand(1)
      y_pred = torch.cat([real_pred, fake_pred])
      loss = torch.mean(loss_discriminator(y_pred.squeeze().double(), y_batch.double()) * torch.tensor(oracle_scores, dtype=torch.float))
              
      D_losses.append(loss.item())
      loss.backward()
      discriminator_optimizer.step()
      #######################################################################
      countergan_optimizer.zero_grad() 
      countergan.set_training_generator(True)
      countergan.generator.train(True)
      countergan.set_training_discriminator(False)
      countergan.discriminator.train(False)
      #######################################################################
      ## Update G network: maximize log(D(G(z)))
      # generator data (fake batch)
      fake_features, fake_edge_index, fake_edge_attr, _ = self.__format_batch(next(generator_loader))
      y_fake = torch.ones((self.batch_size,))
      output = countergan(fake_features, fake_edge_index, fake_edge_attr)
      # calculate the loss
      
      loss = loss_countergan(output.expand(1).double(), y_fake.double())
      loss.backward()
      G_losses.append(loss.item())
      countergan_optimizer.step()
          
      print(f'Iteration {iteration}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
          
      """wandb.log({
        f'iteration_cls={desired_label}': iteration,
        f'loss_d_cls={desired_label}': np.mean(D_losses),
        f'loss_g_cls={desired_label}': np.mean(G_losses)
      })"""
      
      
  def transform_data(self, dataset: Dataset, oracle: Oracle, class_to_explain=0):
    y = torch.from_numpy(np.array([oracle.predict(i) for i in dataset.instances]))
        
    indices = dataset.get_split_indices()[self.fold_id]['train'] 
    y = y[indices]
    
    data_list = []
    for inst in dataset.instances:
      if inst.id in indices:
        data_list.append(inst)

    class_to_explain_indices = (y == class_to_explain).nonzero(as_tuple=True)[0].numpy()
    class_to_not_explain_indices = (y != class_to_explain).nonzero(as_tuple=True)[0].numpy()
    data_list = np.array(data_list, dtype=object)
        
    generator_data = TorchGeometricDataset(data_list[class_to_explain_indices].tolist())
    discriminator_data = TorchGeometricDataset(data_list[class_to_not_explain_indices].tolist())    
    
    #### change in the future to handle more than 1 graph
    self.batch_size = 1
    
    print(generator_data.len())
    print(discriminator_data.len())
    
    generator_loader = DataLoader(generator_data, batch_size=1, shuffle=True, num_workers=2)
    discriminator_loader = DataLoader(discriminator_data, batch_size=1, shuffle=True, num_workers=2)
  
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
    features, edge_index, _ = self.generator(features, edge_index, edge_attr)
    return self.discriminator(features, edge_index, edge_attr)


class Discriminator(nn.Module):

  def __init__(self, n_nodes=28, n_features=1):
    """This class provides a GCN to discriminate between real and generated graph instances"""
    super(Discriminator, self).__init__()

    self.training = False
    self.n_nodes = n_nodes

    self.conv1 = GCNConv(n_features, 64)
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
    edge_probabilities = self.model.decoder.forward_all(encoded_node_features, sigmoid=False)
    edge_probabilities = torch.nan_to_num(edge_probabilities, 0)

    if self.residuals:
      encoded_node_features = torch.add(encoded_node_features, node_features)
      edge_probabilities = rebuild_adj_matrix(len(node_features), edge_list, edge_attr) + edge_probabilities
      edge_probabilities = torch.sigmoid(edge_probabilities)

    return encoded_node_features, edge_list, edge_probabilities
  
class GCNGeneratorEncoder(nn.Module):
  """This class use GCN to generate an embedding of the graph to be used by the generator"""

  def __init__(self, in_channels=1, out_channels=64):
    super().__init__()
    self.conv1 = GCNConv(in_channels, out_channels)
    self.conv2 = GCNConv(out_channels, in_channels)
    
    self.training = False

  def forward(self, x, edge_index, edge_attr):
    x = x.double()
    edge_attr = edge_attr.double()
    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)
    x = F.dropout(x, p=.2, training=self.training)
    x = self.conv2(x, edge_index, edge_attr)
    x = torch.tanh(x)
    return x

  def set_training(self, training):
    self.training = training
    
    
  def __len__(self):
    return len(self.instances)
  
  def __getitem__(self, idx):
    return self.instances[idx]