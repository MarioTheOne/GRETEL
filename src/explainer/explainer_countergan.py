from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle
from src.dataset.data_instance_base import DataInstance

import numpy as np
import json
import pickle
import time
import math
from sklearn.model_selection import train_test_split
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn.functional as F
import torch.nn as nn


class CounteRGANExplainer(Explainer):

    def __init__(self, id, explainer_store_path, n_nodes, batch_size_ratio=0.1, device='cuda', training_iterations=20000, real_label=1, fake_label=0, fold_id=0, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self.name = 'countergan'
        self.batch_size_ratio = batch_size_ratio
        self.batch_size = 1
        self.real_label = real_label # instance label
        self.fake_label = fake_label # CF label
        self.n_nodes = n_nodes
        self.device = device
        self.training_iterations = training_iterations
        self.explainer_store_path = explainer_store_path
        self.fold_id = fold_id
        self._fitted = False
        # self.generator = ResidualGenerator(n_nodes=n_nodes, residuals=True, device=device)
        # self.generator.to(device)
        # self.discriminator = Discriminator(n_nodes, device)
        # self.discriminator.to(device)

        # Creating generator and discriminator for class 0
        self.generator_cl0 = ResidualGenerator(n_nodes=n_nodes, residuals=True, device=device)
        self.generator_cl0.to(device)
        self.discriminator_cl0 = Discriminator(n_nodes, device)
        self.discriminator_cl0.to(device)

        # Creating generator and discriminator for class 1
        self.generator_cl1 = ResidualGenerator(n_nodes=n_nodes, residuals=True, device=device)
        self.generator_cl1.to(device)
        self.discriminator_cl1 = Discriminator(n_nodes, device)
        self.discriminator_cl1.to(device)
        


    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        if(not self._fitted):
            self.fit(oracle, dataset, self.fold_id)

        # Getting the class of the instance
        pred_lbl = oracle.predict(instance)
        cf_instance = instance

        with torch.no_grad():
            torch_data_instance = torch.from_numpy(instance.to_numpy_array())[None, None, :, :]
            torch_data_instance = torch_data_instance.to(torch.float)

            # If the instance belongs to class 1
            if(pred_lbl):
                self.generator_cl0.eval()
                torch_cf = self.generator_cl0(torch_data_instance)
                np_cf = torch_cf.squeeze().cpu().numpy().astype(int)

                cf_instance = DataInstance(-1)
                cf_instance.from_numpy_array(np_cf, store=True)
                
            else: # If the instance belongs to class 0
                self.generator_cl1.eval()
                torch_cf = self.generator_cl1(torch_data_instance)
                np_cf = torch_cf.squeeze().cpu().numpy().astype(int)

                cf_instance = DataInstance(-1)
                cf_instance.from_numpy_array(np_cf, store=True)

            return cf_instance
    

    def save_explainer(self):
        torch.save(self.generator_cl0.state_dict(), os.path.join(self.explainer_store_path, self.name, 'explainer_0'))
        torch.save(self.generator_cl1.state_dict(), os.path.join(self.explainer_store_path, self.name, 'explainer_1'))

    def load_explainer(self):
        self.generator_cl0.load_state_dict(torch.load(os.path.join(self.explainer_store_path, self.name, 'explainer_0')))
        self.generator_cl1.load_state_dict(torch.load(os.path.join(self.explainer_store_path, self.name, 'explainer_1')))


    def fit(self, oracle: Oracle, dataset : Dataset, fold_id=0):
        explainer_name = 'countergan_fit_on_' + dataset.name + '_fold_id_' + str(fold_id)
        explainer_uri = os.path.join(self.explainer_store_path, explainer_name)

        if os.path.exists(explainer_uri):
            # Load the weights of the trained model
            self.name = explainer_name
            self.load_explainer()

        else:
            # Create the folder to store the oracle if it does not exist
            os.mkdir(explainer_uri)        
            self.name = explainer_name

            self._real_fit(self.generator_cl0, self.discriminator_cl0, oracle, dataset, fold_id, real_label=1, fake_label=0)
            self._real_fit(self.generator_cl1, self.discriminator_cl1, oracle, dataset, fold_id, real_label=0, fake_label=1)
        
            # train_loader = self.transform_data(dataset, fold_id)
            # discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
            #                                             lr=0.0002,
            #                                             betas=(0.5, 0.999))
            # generator_optimizer = torch.optim.RMSprop(self.generator.parameters(),
            #                                             lr=4e-4, weight_decay=1e-8)

            # fixed_noise = torch.randn(self.batch_size, self.n_nodes, self.n_nodes, 1, device=self.device)
            
            # loss_bce = nn.BCELoss()
            # loss_nll = nn.NLLLoss()

            # G_losses, D_losses = [], []
            
            # for iteration in range(self.training_iterations):
            #     for i, data in enumerate(train_loader):
            #     # send the data to gpu/cpu
            #         graph, labels = data
            #         graph = graph.to(self.device)
            #         labels = labels.to(self.device)
            #         graph, labels = graph[:,None,:,:], labels[:,None]

            #         #########################
            #         # (1) Update the discriminator network (D). max log(D(x)) + log(1-D(G(z)))
            #         #########################
            #         ## Train with all-real batch
            #         discriminator_optimizer.zero_grad()
            #         self.discriminator.train(True)
            #         self.discriminator.training = True

            #         # Forward pass real batch through the discriminator
            #         output = self.discriminator(graph)
            #         # Calculate loss on all-real batch
            #         errD_real = loss_bce(output, labels)
            #         # Calculate gradients for the discriminator
            #         errD_real.backward()
            #         D_x = output.mean().item()

            #         ## Train with all-fake batch
            #         # Generate batch of latent vectors
            #         noise = torch.randn(self.batch_size, 1, self.n_nodes, self.n_nodes, device=self.device)
            #         # Generate fake graph batch with the generator (G)
            #         fake_graph = self.generator(noise)
            #         fake_labels = torch.full((self.batch_size, 1), self.fake_label,
            #                                 dtype=torch.float, device=self.device)
            #         # Classify all fake batch with the discriminator (D)
            #         output = self.discriminator(fake_graph.detach())
            #         # Calculate D's loss on the all-fake batch
            #         errD_fake = loss_bce(output, fake_labels)
            #         # Calculate the gradients for this batch, accumulated (summed) with previous
            #         # gradients
            #         errD_fake.backward()
            #         D_G_z1 = output.mean().item()
            #         # Compute error of D as sum over the fake and the real batches
            #         errD = errD_real + errD_fake
            #         # Update D
            #         discriminator_optimizer.step()

            #         ##########################
            #         # (2) Update the generator (G) network: maximise log(D(G(z)))
            #         ##########################
            #         generator_optimizer.zero_grad()
            #         self.discriminator.train(False)
            #         self.discriminator.training = False
            #         # fake labels are real for generator cost
            #         fake_labels = torch.full((self.batch_size, 1), self.real_label, dtype=torch.float,
            #                                 device=self.device)
            #         # Since we just updated D, perform another forward pass of all-fake batch
            #         # through D
            #         output = self.discriminator(fake_graph)

            #         # Calculate G's loss based on this output and the counterfactual realism
            #         fake_inst = DataInstance(-1)
            #         fake_inst.from_numpy_array(fake_graph.detach().cpu().numpy().squeeze())

            #         pred_label = oracle.predict(fake_inst)
            #         if not pred_label:
            #             oracle_prediction = torch.from_numpy(np.array([1, 0]))
            #         else:
            #             oracle_prediction = torch.from_numpy(np.array([0, 1]))

            #         oracle_prediction = oracle_prediction.to(torch.float).to(self.device)[None, :]
            #         lbl = labels.to(torch.long)[0]
            #         nll_loss = -loss_nll(oracle_prediction, lbl)

            #         errG = loss_bce(output, fake_labels) + nll_loss  
            #         # Calculate gradients for G
            #         errG.backward()
            #         D_G_z2 = output.mean().item()
            #         # Update G
            #         generator_optimizer.step()

            #         print(f'Iteration [{iteration}/{self.training_iterations}] [{i}]'\
            #                 +f'\tLoss_D: {errD.item()}\tLoss_G: {errG.item()}'\
            #                 +f'\tD(x): {D_x}\tD(G(z)): {D_G_z1} / {D_G_z2}')
                        
            #         G_losses.append(errG.item())
            #         D_losses.append(errD.item())

            # Saving the trained explainer
            self.save_explainer()        

        # setting the flag to signal the explainer was already trained
        self._fitted = True


    def _real_fit(self, generator, discriminator, oracle : Oracle, dataset : Dataset, fold_id, real_label, fake_label):
        train_loader = self.transform_data(dataset, fold_id)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                                    lr=0.0002,
                                                    betas=(0.5, 0.999))
        generator_optimizer = torch.optim.RMSprop(generator.parameters(),
                                                    lr=4e-4, weight_decay=1e-8)
        
        loss_bce = nn.BCELoss()
        loss_nll = nn.NLLLoss()

        G_losses, D_losses = [], []
        
        for iteration in range(self.training_iterations):
            for i, data in enumerate(train_loader):
            # send the data to gpu/cpu
                graph, labels = data
                graph = graph.to(self.device)
                labels = labels.to(self.device)
                graph, labels = graph[:,None,:,:], labels[:,None]

                #########################
                # (1) Update the discriminator network (D). max log(D(x)) + log(1-D(G(z)))
                #########################
                ## Train with all-real batch
                discriminator_optimizer.zero_grad()
                discriminator.train(True)
                discriminator.training = True

                # Forward pass real batch through the discriminator
                output = discriminator(graph)
                # Calculate loss on all-real batch
                errD_real = loss_bce(output, labels)
                # Calculate gradients for the discriminator
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate fake graph batch with the generator (G)
                fake_graph = generator(graph)
                fake_labels = torch.full((self.batch_size, 1), fake_label,
                                        dtype=torch.float, device=self.device)
                # Classify all fake batch with the discriminator (D)
                output = discriminator(fake_graph.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = loss_bce(output, fake_labels)
                # Calculate the gradients for this batch, accumulated (summed) with previous
                # gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                discriminator_optimizer.step()

                ##########################
                # (2) Update the generator (G) network: maximise log(D(G(z)))
                ##########################
                generator_optimizer.zero_grad()
                discriminator.train(False)
                discriminator.training = False
                # fake labels are real for generator cost
                fake_labels = torch.full((self.batch_size, 1), real_label, dtype=torch.float,
                                        device=self.device)
                # Since we just updated D, perform another forward pass of all-fake batch
                # through D
                output = discriminator(fake_graph)

                # Calculate G's loss based on this output and the counterfactual realism
                fake_inst = DataInstance(-1)
                fake_inst.from_numpy_array(fake_graph.detach().cpu().numpy().squeeze())

                pred_label = oracle.predict(fake_inst)
                if not pred_label:
                    oracle_prediction = torch.from_numpy(np.array([1, 0]))
                else:
                    oracle_prediction = torch.from_numpy(np.array([0, 1]))

                oracle_prediction = oracle_prediction.to(torch.float).to(self.device)[None, :]
                lbl = labels.to(torch.long)[0]
                nll_loss = -loss_nll(oracle_prediction, lbl)


                errG = loss_bce(output, fake_labels) + nll_loss   
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                generator_optimizer.step()

                print(f'Iteration [{iteration}/{self.training_iterations}] [{i}]'\
                        +f'\tLoss_D: {errD.item()}\tLoss_G: {errG.item()}'\
                        +f'\tD(x): {D_x}\tD(G(z)): {D_G_z1} / {D_G_z2}')
                    
                G_losses.append(errG.item())
                D_losses.append(errD.item())        
    

    def transform_data(self, dataset: Dataset, fold_id=0):
        X  = np.array([i.to_numpy_array() for i in dataset.instances])
        y = np.array([i.graph_label for i in dataset.instances])

        X_train = X[dataset.get_split_indices()[fold_id]['train']]
        y_train = y[dataset.get_split_indices()[fold_id]['train']]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
        # train_loader = DataLoader(train_dataset, batch_size=int(self.batch_size_ratio*len(X_train)), shuffle=True, num_workers=2, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)

        return train_loader
    


class ResidualGenerator(nn.Module):

    def __init__(self, n_nodes=28, residuals=True, device='cuda'):
        super(ResidualGenerator, self).__init__()

        self.n_nodes = n_nodes
        self.residuals = residuals
        self.device = device

        self.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
        
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
        
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run([3,3], [2,2], [0,0])
        self.fc = nn.Linear(in_features=64 * self.fc_len**2,
                            out_features=128 * (self.n_nodes // 4)**2)
        
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=128,
                                        out_channels=64,
                                        kernel_size=(4,4),
                                        stride=(2,2),
                                        padding=1)
        
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2)

        self.deconv2 = nn.ConvTranspose2d(in_channels=64,
                                        out_channels=64,
                                        kernel_size=(4,4),
                                        stride=(2,2),
                                        padding=1)
        
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.2)

        self.final_conv = nn.Conv2d(in_channels=64,
                                    out_channels=1,
                                    kernel_size=(7,7),
                                    stride=(1,1),
                                    padding=self.__get_same_padding(7,1)
                                    )
        
        
    def init_run(self, kernels, strides, paddings):
        assert len(kernels) == len(strides)
        assert len(kernels) == len(paddings)

        retr = self.n_nodes
        for i in range(len(kernels)):
            retr = int((retr - 2*paddings[i] - kernels[i]) / strides[i]) + 1

        return retr


    def forward(self, graph):
        x = self.conv1(graph)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.leaky_relu3(x)
        x = x.view((-1, 128, self.n_nodes//4, self.n_nodes//4))
        x = self.deconv1(x)
        x = self.leaky_relu4(x)
        x = self.deconv2(x)
        x = self.leaky_relu5(x)
        x = self.final_conv(x)
        x = torch.tanh(x) # Tanh values are in [-1, 1] so allow the residual to add or remove edges

        # building the counterfactual example from the union of the residuals and the original instance
        if self.residuals:
            x = torch.add(graph, x)

        # transforming the output into an adj matrix if the value of a cell is < 0.5 then assign 0 else assign 1
        x = torch.round(x)
        return x


    def __get_same_padding(self, kernel_size, stride):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    


class Discriminator(nn.Module):

    def __init__(self, n_nodes, device='cuda'):
        super(Discriminator, self).__init__()

        self.n_nodes = n_nodes

        self.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2)
                            )
        
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout1 = nn.Dropout2d(p=0.4)

        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
        
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout2 = nn.Dropout2d(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2))
            
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout3 = nn.Dropout2d(p=0.4)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run([3,3,3], [2,2,2], [0,0,0])
        self.fc = nn.Linear(64*self.fc_len**2, 1)

        self.training = False
        self.device = device


    def init_run(self, kernels, strides, paddings):
        assert len(kernels) == len(strides)
        assert len(kernels) == len(paddings)

        retr = self.n_nodes
        for i in range(len(kernels)):
            retr = int((retr - 2*paddings[i] - kernels[i]) / strides[i]) + 1

        return retr


    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise


    def get_same_padding(self, kernel_size, stride, input_dim):
        return 0 if stride != 1 else (kernel_size - stride) // 2


    def forward(self, graph):
        x = self.conv1(graph)
        if self.training:
            x = self.add_gaussian_noise(x)

        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = torch.sigmoid(self.fc(x))

        return x
    
