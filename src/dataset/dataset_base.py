from typing_extensions import Self

from sqlalchemy import false
from src.dataset.instances.base import DataInstance

from abc import ABC, abstractmethod
from typing import Dict, List
import os
import ast
import jsonpickle
import networkx as nx
from sklearn.model_selection import KFold

import torch as th
import dgl
from dgl.data.utils import save_graphs, load_graphs

import numpy as np


class Dataset(ABC):

    def __init__(self, id, config_dict=None) -> None:
        super().__init__()
        self._id = id
        self._name = 'unnamed_dataset'
        self._instance_id_counter = 0
        self._config_dict = config_dict
        self.instances:List[DataInstance] = []
        self._max_n_nodes = 0
        self._n_node_types = 0

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def max_n_nodes(self):
        return self._max_n_nodes

    @max_n_nodes.setter
    def max_n_nodes(self, new_val):
        self._max_n_nodes = new_val

    @property
    def n_node_types(self):
        return self._n_node_types

    @n_node_types.setter
    def n_node_types(self, new_val):
        self._n_node_types = new_val

    def write_data(self, datasets_folder_path, graph_format='edge_list'):
        """Writes the dataset into files inside a given folder
        -------------
        INPUT:
            instances    : the instance in the dataset to write 
            datasets_folder_path : the path to the folder where the dataset are store, 
            a folder will be created inside it to store this dataset

        -------------
        OUTPUT:
            void
        """
        # Creating a folder to contain the dataset
        dataset_path = os.path.join(datasets_folder_path, self._name)
        os.mkdir(dataset_path)

        # Creating a file to contain the name of the dataset
        with open(os.path.join(dataset_path, 'dataset_name.txt'), 'w') as ds_name_writer:
            ds_name_writer.write(self._name)

        # Creating a file to contain the id of the dataset
        if self.id is not None:
            with open(os.path.join(dataset_path, 'dataset_id.json'), 'w') as ds_id_writer:
                ds_id_writer.write(jsonpickle.encode(self._id))

        # Creating a file to contain the name of each instance
        f_gnames = open(os.path.join(dataset_path, 'graph_names.txt'), 'a')

        # Saving each instance
        for i in self.instances:
            # Writing the name of the instance into graph_names.txt
            i_name = i.name
            f_gnames.writelines(i_name + '\n')

            # Creating a folder to contain the files associated with the instance
            i_path = os.path.join(dataset_path, i_name)
            os.mkdir(i_path)

            if graph_format == 'edge_list':
                # Writing the instance graph into edgelist format
                i_graph_path = os.path.join(i_path, i_name + '_graph.edgelist')
                nx.write_edgelist(i.graph, i_graph_path)

            elif graph_format == 'adj_matrix':
                # Writing the instance graph into adj_matrix format
                i_graph_path = os.path.join(i_path, i_name + '_graph.adjlist')
                nx.write_multiline_adjlist(i.graph, i_graph_path)

            elif graph_format == 'dgl':
                # Write the dgl graph into file
                if i.graph_dgl is not None:
                    i_graph_path = os.path.join(i_path, i_name + '_graph_dgl.bin')
                    save_graphs(i_graph_path, [i.graph_dgl], {"labels" : th.Tensor([i.graph_label])})
            else:
                raise ValueError('The chosen graph format is not supported')

            # Writing the node labels into file in json format
            if i.node_labels is not None:
                with open(os.path.join(i_path, i_name + '_node_labels.json'), 'w') as node_labels_writer:
                    node_labels_writer.write(jsonpickle.encode(i.node_labels))

                    # Writing the edge labels into file in json format
            if i.edge_labels is not None:
                with open(os.path.join(i_path, i_name + '_edge_labels.json'), 'w') as edge_labels_writer:
                    edge_labels_writer.write(jsonpickle.encode(i.edge_labels))

            # Writing the graph label into file in json format
            if i.graph_label is not None:
                with open(os.path.join(i_path, i_name + '_graph_label.json'), 'w') as graph_label_writer:
                    graph_label_writer.write(jsonpickle.encode(i.graph_label))

            # Writing the minimal counterfactual distance into file in json format
            if i.minimum_counterfactual_distance is not None:
                with open(os.path.join(i_path, i_name + '_mcd.json'), 'w') as mcd_writer:
                    mcd_writer.write(jsonpickle.encode(i.minimum_counterfactual_distance))

        # Writing the splits into file in json format
        if self.splits is not None:
            with open(os.path.join(i_path, i_name + '_splits.json'), 'w') as split_writer:
                split_writer.write(jsonpickle.encode(self.splits))

        f_gnames.close()

    def read_data(self, dataset_path, graph_format='edge_list'):
        """Reads the dataset from files inside a given folder
        -------------
        INPUT:
            dataset_path : the path to the folder containing the dataset
        -------------
        OUTPUT:
            A list of instances (dictionaries) containing the graphs, labels, and 
            minimum counterfactual distance
        """

        # Reading the name of the dataset from file
        with open(os.path.join(dataset_path, 'dataset_name.txt'), 'r') as ds_name_reader:
            self._name = ds_name_reader.read()

        # Reading the id of the dataset from file
        dataset_id_uri = os.path.join(dataset_path, 'dataset_id.json')
        if os.path.exists(dataset_id_uri):
            with open(dataset_id_uri, 'r') as ds_id_reader:
                str_id = jsonpickle.decode(ds_id_reader.read())
                self._id = str_id

        # Reading the file containing the name of each instance
        f_gnames = open(os.path.join(dataset_path, 'graph_names.txt'), 'r')

        result = []
        instance_number = 0
        # Iterate over each instance and load them
        for line in f_gnames.readlines():
            inst = DataInstance(id=instance_number)
            instance_number += 1

            # Getting the instance name and storing the path to the instance folder
            i_name = str.strip(line, '\n')
            inst.name = i_name
            i_path = os.path.join(dataset_path, i_name)

            if graph_format == 'edge_list':
                # Reading the graph from the edgelist
                i_path_graph = os.path.join(i_path, i_name + '_graph.edgelist')
                # If this line is removed the keys of the nodes are casted to str
                g = nx.read_edgelist(i_path_graph, nodetype=int)
                inst.graph = g
            elif graph_format == 'adj_matrix':
                # Reading the graph from the edgelist
                i_path_graph = os.path.join(i_path, i_name + '_graph.adjlist')
                # If this line is removed the keys of the nodes are casted to str
                g = nx.read_multiline_adjlist(i_path_graph, nodetype=int)
                inst.graph = g
            elif graph_format == 'dgl':
                # Reading the graph from the dgl dump
                i_graph_path = os.path.join(i_path, i_name + '_graph_dgl.bin')
                g = load_graphs(i_path_graph)[0]
                inst.graph_dgl = g
                inst.graph = dgl.to_networkx(g)
            else:
                raise ValueError('The chosen graph format is not supported')

            # Reading the node labels from json file
            node_labels_uri = os.path.join(i_path, i_name + '_node_labels.json')
            if os.path.exists(node_labels_uri):
                with open(node_labels_uri, 'r') as node_labels_reader:
                    str_dict = jsonpickle.decode(node_labels_reader.read())
                    node_labels = {}
                    for k, v in str_dict.items():
                        node_labels[ast.literal_eval(k)] = v
                    inst.node_labels = node_labels

            # Reading the node labels from json file
            node_labels_uri = os.path.join(i_path, i_name + '_node_labels.json')
            if os.path.exists(node_labels_uri):
                with open(node_labels_uri, 'r') as node_labels_reader:
                    str_dict = jsonpickle.decode(node_labels_reader.read())
                    node_labels = {}
                    for k, v in str_dict.items():
                        node_labels[ast.literal_eval(k)] = v
                    inst.node_labels = node_labels

            # Reading the edge labels from json file
            edge_labels_uri = os.path.join(i_path, i_name + '_edge_labels.json')
            if os.path.exists(edge_labels_uri):
                with open(edge_labels_uri, 'r') as edge_labels_reader:
                    str_dict = jsonpickle.decode(edge_labels_reader.read())
                    edge_labels = {}
                    for k, v in str_dict.items():
                        edge_labels[ast.literal_eval(k)] = v
                    inst.edge_labels = edge_labels

                    # Reading the graph label from json file
            graph_label_uri = os.path.join(i_path, i_name + '_graph_label.json')
            if os.path.exists(graph_label_uri):
                with open(graph_label_uri, 'r') as graph_label_reader:
                    inst.graph_label = jsonpickle.decode(graph_label_reader.read())

            # Reading the minimum counterfactual distance from json file
            mcd_uri = os.path.join(i_path, i_name + '_mcd.json')
            if os.path.exists(mcd_uri):
                with open(mcd_uri, 'r') as mcd_reader:
                    inst.minimum_counterfactual_distance = jsonpickle.decode(mcd_reader.read())

            result.append(inst)

        # Reading the splits of the dataset
        splits_uri = os.path.join(i_path, i_name + '_splits.json')
        if os.path.exists(splits_uri):
            with open(splits_uri, 'r') as split_reader:
                sp = jsonpickle.decode(split_reader.read())
                self.splits = sp

        f_gnames.close()
        self.instances = result

    def get_data(self):
        """Return the list of all data instances
        -------------
        OUTPUT:
            A list containing the dictionaries of all data instances
        """
        return self.instances
    

    def get_instance(self, i):
        """Returns the data instance at the i-th position'
        -------------
        INPUT:
            i : integer representing the position of the instance in the dataset
        -------------
        OUTPUT:
            A dictionary representing the i-th data instance in the data set 
        """
        return self.instances[i]

    def get_data_len(self):
        """Returns the number of data instances in the data set'
        -------------
        OUTPUT:
            An integer representing the number of instances in the data set 
        """
        return len(self.instances)

    def get_split_indices(self):
        """Returns a list of dictionaries containing the splits of the instance indices into 'test' and 'train'
        -------------
        OUTPUT:
            A dictionary containing the data splits 
        -------------
        EXAMPLES:
            >>> from dataset_base import Dataset
            >>> ds = Dataset()
            >>> ds.read_data('data_path')
            >>> ds.generate_splits()
            >>> print(ds.get_split_indices()[0]['train'])
                [1,3,5]
        """
        return self.splits
    

    def generate_splits(self, n_splits=10, shuffle=True):
        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        self.splits = []
        spl = kf.split([i for i in range(0, len(self.instances))], 
            [g.graph_label for g in self.instances])

        for train_index, test_index in spl:
            self.splits.append({'train': train_index, 'test': test_index})

    
    def load_or_generate_splits(self, dataset_folder, n_splits=10, shuffle=True):

         # Reading the splits of the dataset
        splits_uri = os.path.join(dataset_folder, 'splits.json')
        if os.path.exists(splits_uri):
            with open(splits_uri, 'r') as split_reader:
                sp = jsonpickle.decode(split_reader.read())
                self.splits = sp
        else:
            self.generate_splits(n_splits=n_splits, shuffle=shuffle)
            with open(os.path.join(dataset_folder, 'splits.json'), 'w') as split_writer:
                split_writer.write(jsonpickle.encode(self.splits))
        


    def gen_tf_data(self):
        for i in self.instances:
            graph = i.to_numpy_arrays(false)
            activity = i.graph_label
            yield graph, activity

    def num_classes(self):
        return len(np.unique([i.graph_label for i in self.instances]))