import networkx as nx
import numpy as np

from src.n_dataset.generators.base import Generator
from src.n_dataset.instances.graph import GraphInstance


class TreeCycles(Generator):
    
    def init(self):
        self.num_cycles = self.local_config['parameters'].get('num_cycles', 5)
        self.num_instances = self.local_config['parameters'].get('num_instances', 1000)
        self.num_nodes_per_instance = self.local_config['parameters'].get('num_nodes_per_instance', 300)
        self.num_nodes_in_cycles = self.local_config['parameters'].get('num_nodes_in_cycles', 200)
        self.generate_dataset()
        
    def generate_dataset(self):
        # Creating the empty list of instances
        for i in range(self.num_instances):
            # Randomly determine if the graph is going to contain cycles or just be a tree
            has_cycles = np.random.randint(0,2)

            # If the graph will contain cycles
            if(has_cycles):
                # Choose randomly the number of nodes in each cycle and the number of cycle subgraphs
                num_cycles = np.random.randint(low=1, high=66)
                cycle_size = np.random.randint(low=3, high=20)
                # Check that the total number is lower than the maximum amount of nodes in cycles
                while(num_cycles*cycle_size > self.num_nodes_in_cycles):
                    num_cycles = np.random.randint(low=1, high=66)
                    cycle_size = np.random.randint(low=3, high=20)

                # Calculating how many nodes have to contain the tree to maintain the total
                # number of nodes per instance
                n_tree = self.num_nodes_per_instance - (num_cycles*cycle_size)
                # Creating the tree-cycles graph
                tc_graph, tc_node_labels, tc_edge_labels = self.create_tree_cycles_graph(n_nodes_tree=n_tree,
                                                                                         cycle_size=cycle_size,
                                                                                         num_cycles=num_cycles)
                # Creating the instance
                self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(tc_graph), label=1))
            else:
                # Generating a random tree containing all the nodes of the instance
                tc_graph, tc_node_labels, tc_edge_labels = self.create_tree(tree_size=self.num_instances, role_label=0)
                self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(tc_graph), label=0))

            self.context.logger.info("Generated instance with id:"+str(i))
        
    
    def get_num_instances(self):
        return len(self.dataset.instances)
        
    def create_tree(self, tree_size, role_label=0):
        """Builds a random tree graph
        -------------
        INPUT:
        tree_size   :    int the number of nodes in the tree
        role_label  :    int the label to be assigned to the nodes of the graph
        -------------
        OUTPUT:
        graph       :    an undirected acyclic graph
        node_labels :    dict containing the role labels of the nodes
        edge_labels :    dict containing the role labels of the edges
        """
        graph = nx.random_tree(n=tree_size)

        # Creating the dictionary containing the node labels 
        node_labels = {}
        for n in graph.nodes:
            node_labels[n] = role_label

        # Creating the dictionary containing the edge labels
        edge_labels = {}
        for e in graph.edges:
            edge_labels[e] = role_label

        return graph, node_labels, edge_labels
    
    def create_cycle(self, start, cycle_size, role_label=1):
        """Builds a cycle graph, with index of nodes starting at start
        and role_ids at role_start
        -------------
        INPUT:
        start       :    int starting index for the shape
        role_label  :    int the role label for the nodes in the shape
        -------------
        OUTPUT:
        graph       :    a house shape graph, with ids beginning at start
        role_labels :    list of the roles of the nodes
        """

        # Creating an empty graph and adding the nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(start, start + cycle_size))

        # Adding the edges  of the graph
        for i in range(cycle_size - 1):
            graph.add_edges_from([(start + i, start + i + 1)])

        graph.add_edges_from([(start + cycle_size - 1, start)])
        
        # Creating the dictionary containing the node labels 
        node_labels = {}
        for n in graph.nodes:
            node_labels[n] = role_label

        # Creating the dictionary containing the edge labels
        edge_labels = {}
        for e in graph.edges:
            edge_labels[e] = role_label

        # Returning the cycle graph and the role labels
        return graph, node_labels, edge_labels
    
    
    def join_graph(self, G1, G2, num_add_edges, G1_node_labels, G2_node_labels, G1_edge_labels, G2_edge_labels):
        """ Join two graphs along matching nodes, then add edges between the nodes of both subgraphs.
        -------------
        INPUT:
            G1, G2        : Networkx graphs to be joined.
            num_add_edges : number of edges to be added connecting the two subgraphs.
            G1_node_labels: The role labels of the nodes of G1
            G2_node_labels: The role labels of the nodes of G2
            G1_edge_labels: The role labels of the edges of G1
            G2_edge_labels: The role labels of the edges of G2
        -------------
        OUTPUT:
            merged_graph      : A new graph, result of merging and perturbing G1 and G2.
            merged_node_labels: the node labels of the combined graph
            merged_edge_labels: the edge labels of the combined graph
        """

        # Joining the two graphs
        G_result = nx.compose(G1, G2)

        # Merging the labels of G2 into those of G1
        result_node_labels = {**G1_node_labels, **G2_node_labels}
        result_edge_labels = {**G1_edge_labels, **G2_edge_labels}
        
        # Randomly adding edges between the nodes of the two subgraphs
        edge_cnt = 0
        while edge_cnt < num_add_edges:
            # Choosing the two node to connect
            node_1 = np.random.choice(G1.nodes())
            node_2 = np.random.choice(G2.nodes())
            
            # Creating an edge between the nodes
            G_result.add_edge(node_1, node_2)
            result_edge_labels[(node_1, node_2)] = 0

            edge_cnt += 1

        return G_result, result_node_labels, result_edge_labels


    def create_tree_cycles_graph(self, n_nodes_tree: int, cycle_size: int, num_cycles: int):
        """ Generate a graph composed by a base tree with added cycles connected to the tree by one edge
        -------------
        INPUT:
            n_nodes_tree  : number of nodes of the base tree graph
            cycle_size    : number of nodes in each cycle
            num_cycles      : the number of cycle graphs to add to the base tree
        -------------
        OUTPUT:
            tree-cycle_graph: A base tree graph with some cycle subgraphs connected to it.
            tree-cycle_node_labels: the node labels of the tree-cycle graph
            tree-cycle_edge_labels: the edge labels of the tree-cycle graph
        """

        # Generate the base tree graph
        result_graph, result_node_labels, result_edge_labels = self.create_tree(tree_size=n_nodes_tree, role_label=0)

        # Generate the cycle graphs and connect them to the tree
        for _ in range(0, num_cycles):
            c_graph, c_node_labels, c_edge_labels = self.create_cycle(start=len(result_node_labels), cycle_size=cycle_size, role_label=1)

            result_graph, result_node_labels, result_edge_labels = self.join_graph(G1=result_graph, G2=c_graph, 
                num_add_edges=1, G1_node_labels=result_node_labels, G2_node_labels=c_node_labels,
                G1_edge_labels=result_edge_labels, G2_edge_labels=c_edge_labels)

        # Return the tree-cycle graph
        return result_graph, result_node_labels, result_edge_labels