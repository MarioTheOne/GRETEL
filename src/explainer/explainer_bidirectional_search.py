from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle
from src.dataset.data_instance_base import DataInstance

from abc import abstractmethod
import os
import numpy as np
import random
import networkx as nx
import sys

class BidirectionalHeuristicExplainerBase(Explainer):
    """
    A generic implementation of the Counterfactual Bidirectional Explainer proposed in the paper 
    "Abrate, Carlo, and Francesco Bonchi. "Counterfactual Graphs for Explainable Classification of 
    Brain Networks." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.
    2021."
    """

    # Begin> Implemented by us ///////////////////////////////////////////////////////////////////////////////////////

    def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self._gd = instance_distance_function
        self._name = 'AbstractBidirectionalSearch'


    @abstractmethod
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        pass


    def data_search(self, instance, oracle: Oracle, dataset: Dataset):
        l_input_inst = oracle.predict(instance)
        oracle_calls = 1

        min_counterfactual = None
        min_counterfactual_dist = sys.float_info.max

        for d_inst in dataset.instances:

            l_data_inst = oracle.predict(d_inst)
            oracle_calls += 1

            if ( (instance.name != d_inst.name) and (l_input_inst != l_data_inst) ):
                d_inst_dist = self._gd.evaluate(instance, d_inst, oracle)

                if (d_inst_dist < min_counterfactual_dist):
                    min_counterfactual_dist = d_inst_dist
                    min_counterfactual = d_inst
        
        return min_counterfactual.to_numpy_array(), oracle_calls

    # End> Implemented by us ///////////////////////////////////////////////////////////////////////////////////////

    # Begin> 3rd party adapted /////////////////////////////////////////////////////////////////////////////////////

    def _edit_distance(self, g_1,g_2):
        '''Returns a particular version of Graph Edit Distance when only edge changes are considered and
        the graphs are already matched
        '''
        return self._tot_edges(abs(g_1-g_2))


    def _bernoulli(self, p):
        ''' p is the probability of removing an edge.
        '''
        return True if random.random() < p else False


    def _get_change_list(self, g1,g2):
        edges = []
        g_diff = abs(g1-g2)
        dim_g = len(g1)
        for i in range(dim_g):
            for j in range(i,dim_g):
                if g_diff[i][j]==1:
                    edges.append((i,j))
        return edges


    def _tot_edges(self, g):
        '''Returns the total number of edges for undirected graphs
        '''
        return sum([sum(el) for el in g])/2

    # End> 3rd party adapted /////////////////////////////////////////////////////////////////////////////////////


class ObliviousBidirectionalSearchExplainer(BidirectionalHeuristicExplainerBase):
    """
    An implementation of the Counterfactual Explainer proposed in the paper "Abrate, Carlo, and Francesco Bonchi. 
    "Counterfactual Graphs for Explainable Classification of Brain Networks." 
    Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021."
    """

    # Made by us/////////////////////////////////////////////////////////////////////////////////////
    def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, instance_distance_function, config_dict)
        self._name = 'Oblivious_Bidirectional_Search'


    def explain(self, instance : DataInstance, oracle: Oracle, dataset: Dataset):

        # Get the label of the original instance
        l_inst = oracle.predict(instance)
        # Get the label of the counterfactual (just for binary classification problems)
        l_counterfactual = 1 - l_inst

        instance_matrix = instance.to_numpy_array()
        # Try to get a first counterfactual with the greedy "Oblivious Forward Search"
        ged, counterfactual, oracle_calls = self.oblivious_forward_search(oracle,
                                                                          instance, 
                                                                          instance_matrix,
                                                                          l_counterfactual)

        final_counterfactual, edit_distance, oracle_calls, info = self.oblivious_backward_search(oracle,
                                                                                        instance, 
                                                                                        instance_matrix, 
                                                                                        counterfactual, 
                                                                                        l_counterfactual)

        # Converting the final counterfactual into a DataInstance
        result = DataInstance(-1)
        result.from_numpy_array(final_counterfactual)
        return result

    
    

    # ///////////////////////////////////////////////////////////////////////////////////////////////

    # 3rd party adapted /////////////////////////////////////////////////////////////////////////////

        
    def oblivious_forward_search(self, oracle: Oracle, instance : DataInstance, g_o, y_bar, k=5, lambda_g=2000, p_0=0.5):
        '''
        Oblivious Forward Search as implemented by Abrate and Bonchi
        '''
        dim = len(g_o)
        l=0
        
        # Candidate counterfactual
        g_c = np.copy(g_o)
        r = abs(1-y_bar)

        # Create add and remove sets of edges
        g_add = []
        g_rem = []
        for i in range(dim):
            for j in range(i,dim):
                if i!=j:
                    if g_c[i][j]>0.5: # Add existing edges to the remove list
                        g_rem.append((i,j))
                    else:
                        g_add.append((i,j)) # Add non-exisitng edges to the add list

        # randomize and remove duplicate
        random.shuffle(g_add)
        random.shuffle(g_rem)
        
        # Start the search
        while(l<lambda_g): # While the maximum number of oracle calls is not exceeded
            ki=0
            while(ki<k): # Made a number of changes (edge adds/removals) no more than k
                if self._bernoulli(p_0):
                    # Added code //////////////////////////////////////////
                    if (len(g_rem) > 0):
                    # /////////////////////////////////////////////////////
                        # remove
                        i,j = g_rem.pop(0)
                        g_c[i][j]=0
                        g_c[j][i]=0
                        g_add.append((i,j))
                        #random.shuffle(g_add)
                        # added code /////////////////////////////////////////
                        ki+=1
                        # ////////////////////////////////////////////////////
                else:
                    # Added code //////////////////////////////////////////
                    if (len(g_add) > 0):
                    # /////////////////////////////////////////////////////
                        # add
                        i,j = g_add.pop(0)
                        g_c[i][j]=1
                        g_c[j][i]=1
                        g_rem.append((i,j))
                        #random.shuffle(g_rem)
                        # added code /////////////////////////////////////////
                        ki+=1
                        # ////////////////////////////////////////////////////
                # modified code /////////////////////////////////////////
                # ki+=1
                # ///////////////////////////////////////////////////////
            ki=0

            # original call r = oracle(g_c) /////////////////////////////////////////////
            inst = DataInstance(-1)
            inst.from_numpy_array(g_c)
            # inst.max_n_nodes = instance.max_n_nodes
            # inst.n_node_types = instance.n_node_types
            r = oracle.predict(inst)
            # ///////////////////////////////////////////////////////////////////////////
            l += 1 # Increase the oracle calls counter

            if r==y_bar:
                #print('- A counterfactual is found!')
                d = self._edit_distance(g_o,g_c)
                return d,g_c,l
            if len(g_rem)<1:
                print('no more remove')
        
        # m-comment: return the original graph if no counterfactual was found
        return 0, g_o, l


    def oblivious_backward_search(self, oracle: Oracle, instance : DataInstance, g ,gc1 ,y_bar ,k=5 , l_max=2000):
        '''
        '''
        # no info list declared inside the function in the original code ///////////////////////
        info = []
        # //////////////////////////////////////////////////////////////////////////////////////

        gc = np.copy(gc1)
        edges = self._get_change_list(g,gc)
        d = self._edit_distance(g,gc)
        random.shuffle(edges)
        li=0
        while(li<l_max and len(edges)>0 and d>1):
            ki = min(k,len(edges))
            gci = np.copy(gc)
            edges_i = [edges.pop(0) for i in range(ki)]
            for i,j in edges_i:
                if gci[i][j]>0.5:
                    gci[i][j] = 0
                    gci[j][i] = 0
                else:
                    gci[i][j] = 1
                    gci[j][i] = 1

            # Oringinal call r = oracle(gci) /////////////////////////////////////////////////////
            inst = DataInstance(-1)
            inst.from_numpy_array(gci)
            # inst.max_n_nodes = instance.max_n_nodes
            # inst.n_node_types = instance.n_node_types
            r = oracle.predict(inst)
            # ////////////////////////////////////////////////////////////////////////////////////
            li += 1

            if r==y_bar:
                gc = np.copy(gci)
                d = self._edit_distance(g,gc)
                #print('ok --> ',r,d,l,k)
                info.append((r,d,li,ki))
                k+=1
            else:
                d = self._edit_distance(g,gc)
                info.append((r,d,li,ki))

                if k>1:
                    k-=1
                    edges = edges + edges_i

        return gc, self._edit_distance(g,gc), li, info


class DataDrivenBidirectionalSearchExplainer(BidirectionalHeuristicExplainerBase):
    """
    An implementation of the Counterfactual Data-Driven Bidirectional Search Explainer proposed in the paper "Abrate, Carlo, and Francesco Bonchi. 
    "Counterfactual Graphs for Explainable Classification of Brain Networks." 
    Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021."
    """

    # Begin> Made by us /////////////////////////////////////////////////////////////////////////////////////
    def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, instance_distance_function, config_dict)
        self._name = 'Data-Driven_Bidirectional_Search'


    def explain(self, instance : DataInstance, oracle: Oracle, dataset: Dataset):
        """
        Uses a combination of Data-driven Forward Search with Data-driven Backward Search as described in "Abrate, Carlo, and Francesco Bonchi. 
        "Counterfactual Graphs for Explainable Classification of Brain Networks." Proceedings of the 27th ACM SIGKDD Conference on Knowledge
        Discovery & Data Mining. 2021."
        """
        edges_prob = self.get_edge_probabilities(dataset, oracle)
        instance_label = oracle.predict(instance)

        # this is only true for binary classification problems
        counterfactual_label = 1 - instance_label
        original_graph = instance.to_numpy_array()

        ged, counterfactual, oracle_calls = self.DFS(oracle, original_graph, counterfactual_label, edges_prob)

        final_counterfactual, ged, oracle_calls, info = self.bb_prob_2(oracle, original_graph, counterfactual, counterfactual_label, edges_prob)

        result = DataInstance(-1)
        result.from_numpy_array(final_counterfactual)
        return result


    def data_driven_forward_search(self, instance : DataInstance, oracle: Oracle, dataset: Dataset) -> DataInstance:
        """
        This method calls the functions of the Data-driven Forward Search in the right order and making the necessary conversions
        """
        edges_prob = self.get_edge_probabilities(dataset, oracle)
        instance_label = oracle.predict(instance)

        # this is only true for binary classification problems
        counterfactual_label = 1 - instance_label

        ged, counterfactual, oracle_calls = self.DFS(oracle, instance.to_numpy_array(), counterfactual_label, edges_prob)

        result = DataInstance(-1)
        result.from_numpy_array(counterfactual)
        return result


    def data_driven_forward_search2(self, instance : DataInstance, oracle: Oracle, dataset: Dataset) -> DataInstance:
        """
        This method calls the functions of the Data-driven Forward Search (Forward Probabilistic version) in the right order and making the necessary conversions
        """

        edges_prob = self.get_edge_probabilities2(dataset, oracle)
        dim_g = len(instance.graph.nodes)
        instance_label = oracle.predict(instance)

        # this is only true for binary classification problems
        counterfactual_label = 1 - instance_label

        ged, counterfactual, oracle_calls = self.forward_probabilistic(oracle, instance.to_numpy_array(), counterfactual_label, edges_prob, dim_g)

        result = DataInstance(-1)
        result.from_numpy_array(counterfactual)
        return result


    def data_driven_backward_search(self, instance : DataInstance, counterfactual: DataInstance, oracle: Oracle, dataset: Dataset) -> DataInstance:
        edges_prob = self.get_edge_probabilities(dataset, oracle)
        instance_label = oracle.predict(instance)

        # this is only true for binary classification problems
        counterfactual_label = 1 - instance_label

        final_counterfactual, ged, oracle_calls, info = self.bb_prob_2(oracle, instance.to_numpy_array(), counterfactual.to_numpy_array(), counterfactual_label, edges_prob)

        result = DataInstance(-1)
        result.from_numpy_array(final_counterfactual)
        return result

    # End> Made by us ///////////////////////////////////////////////////////////////////////////////////////

    # Begin> Adapted 3rd party //////////////////////////////////////////////////////////////////////////////

    def get_edge_probabilities(self, dataset: Dataset, oracle: Oracle):
        """
        This method is meant to be used with DFS
        """
        # Nodes frequency
        #create the two matices as the count of the frequency of each edge to be in a graph of the dataset

        # Modified code /////////////////////////////////////////////////
        # dim_g = 116
        dim_g = len(dataset.instances[0].graph.nodes)
        # ///////////////////////////////////////////////////////////////

        g_0 = np.zeros((dim_g,dim_g))
        g_1 = np.zeros((dim_g,dim_g))

        # Modified code //////////////////////////////////////////////////
        # for k,v in data.items():
        #     g = v[1]
        #     y_hat = oracle(g)
        #     if y_hat==0:
        #         g_0 = np.add(g_0,g)
        #     else:
        #         g_1 = np.add(g_1,g)

        for inst in dataset.instances:
            g = inst.to_numpy_array()
            y_hat = oracle.predict(inst)
            if y_hat==0:
                g_0 = np.add(g_0,g)
            else:
                print("here", g)
                print("here2", g_1)
                g_1 = np.add(g_1,g)

        # //////////////////////////////////////////////////

        g_01 = g_0-g_1
        g_10 = g_1-g_0

        min_01 = g_01.min()
        max_01 = g_01.max()
        g01 = np.ones((dim_g,dim_g))+(g_01-min_01)/(max_01-min_01)

        min_10 = g_10.min()
        max_10 = g_10.max()
        g10 = np.ones((dim_g,dim_g))+(g_10-min_10)/(max_10-min_10)

        prob_initial = {0:g01/g01.sum(),1:g10/g10.sum()}

        g00 = np.ones((dim_g,dim_g))
        uniform_initial_prop = {0:g00/g00.sum(),1:g00/g00.sum()}

        # Added code ///////////////////////////////////////////
        # it is based on the call they perform
        # d_final,g_c_final,lambda_final = DFS(g,y_bar,prob_initial)
        return prob_initial


    def DFS_select(self, g,edges, y_bar, ki, edges_prob, p_0=0.5):
        '''
        '''
        edges_prob_rem = np.array([])
        edges_prob_add = np.array([])
        edges_add = []
        edges_rem = []
        e = []
        dim = len(g)
        for i in range(dim):
            for j in range(dim):
                if (i,j) not in edges:
                    if g[i][j]>0:
                        edges_prob_rem = np.append(edges_prob_rem,edges_prob[1-y_bar][i][j])
                        edges_rem.append((i,j))
                    else:
                        edges_prob_add = np.append(edges_prob_add,edges_prob[y_bar][i][j])
                        edges_add.append((i,j))
        edges_prob_add = edges_prob_add/edges_prob_add.sum()
        edges_prob_rem = edges_prob_rem/edges_prob_rem.sum()
        #print('-- ',len(edges_rem),len(edges_add),len(edges))
        edges_i = []
        kii=0
        while(kii<ki):
            kii+=1
            if self._bernoulli(p_0) and len(edges_add)>0:
                #add
                n = np.random.choice(range(len(edges_add)), size=1, p=edges_prob_add)[0]
                i,j = edges_add[n]
                g[i][j]=1
                g[j][i]=1
            elif len(edges_rem)>0:
                #remove
                n = np.random.choice(range(len(edges_rem)), size=1, p=edges_prob_rem)[0]
                i,j = edges_rem[n]
                g[i][j]=0
                g[j][i]=0
            edges.append((i,j))
        return g,edges


    def DFS(self, oracle: Oracle, g, y_bar, edges_prob, k=10, l_max=2000):
        '''
        '''
        info = []
        gc = np.copy(g)
        d = self._edit_distance(g,gc)
        li=0
        edges=[]
        while(li<l_max):
            gc,edges = self.DFS_select(gc,edges,y_bar,k,edges_prob,)

            # Modified code ////////////////////////////////////////////////
            # r = oracle(gc)
            inst = DataInstance(-1)
            inst.from_numpy_array(gc)
            r = oracle.predict(inst)
            # //////////////////////////////////////////////////////////////
            #print(li,len(edges),r)
            li += 1
            if r==y_bar:
                #print('- A counterfactual is found!')

                # modified code because g_0 does not exist in the current scope ////////////////////////
                # d = self._edit_distance(g_o, gc)
                d = self._edit_distance(g, gc)
                # //////////////////////////////////////////////////////////////////////////////////////

                # modified code because l does not exist in the current scope //////////////////////////
                # return d, gc, l
                return d, gc, li
                # //////////////////////////////////////////////////////////////////////////////////////

        # modified code because l does not exist in the current scope //////////////////////////
        # return 0, gc, l
        return 0, gc, li


    def get_edge_probabilities2(self, dataset: Dataset, oracle: Oracle):
        """
        This method is meant to be used with forward_probabilistic
        """
        # Modified code /////////////////////////////////////////////////
        # dim_g = 116
        dim_g = len(dataset.instances[0].graph.nodes)
        # ///////////////////////////////////////////////////////////////

        g_0 = np.ones((dim_g,dim_g))
        g_1 = np.ones((dim_g,dim_g))

        # Modified code //////////////////////////////////////////////////
        # for k,v in data.items():
        #     g = v[1]
        #     y_hat = oracle(g)
        #     if y_hat==0:
        #         g_0 = np.add(g_0,g)
        #     else:
        #         g_1 = np.add(g_1,g)

        for inst in dataset.instances:
            g = inst.to_numpy_array()
            y_hat = oracle.predict(inst)
            if y_hat==0:
                g_0 = np.add(g_0,g)
            else:
                g_1 = np.add(g_1,g)

        # //////////////////////////////////////////////////

        prob_initial = {0:g_0/g_0.sum(),1:g_1/g_1.sum()}

        ## Edges probabilities
        # Nodes Class 0
        nodes_0_sum = np.array([sum(el) for el in g_0]).sum()
        edges_0 = g_0.ravel()/nodes_0_sum

        # Nodes Class 1
        nodes_1_sum = np.array([sum(el) for el in g_1]).sum()
        edges_1 = g_1.ravel()/nodes_1_sum

        #edges = np.array(edges)
        edges_prob = {0:edges_0, 1:edges_1}

        # Added code to return the edges_prob ///////////////////////
        return edges_prob


    def forward_probabilistic(self, oracle: Oracle, g_o, y_bar, edges_prob, dim_g, lambda_g=2000, p_0=0.5):
        '''
        '''
        dim = len(g_o)
        edges = []
        e = []
        k = 0
        for i in range(dim_g):
            for j in range(dim_g):
                edges.append((i,j))
                e.append(k)
                k+=1
        l=0
        
        # Candidate counterfactual
        g_c = np.copy(g_o)
        r = abs(1-y_bar)
        
        # Start the search
        while(l<lambda_g):
            if self._bernoulli(p_0):
                # remove
                n = np.random.choice(e, size=1, p=edges_prob[abs(y_bar-1)])[0]
                i,j = edges[n]
                g_c[i][j]=0
                g_c[j][i]=0
            else:
                # add
                n = np.random.choice(e, size=1, p=edges_prob[y_bar])[0]
                i,j = edges[n]
                g_c[i][j]=1
                g_c[j][i]=1

            # Modified code ////////////////////////////////////////////////////////////////////////
            # r = oracle(g_c)
            inst = DataInstance(-1)
            inst.from_numpy_array(g_c)
            r = oracle.predict(inst)
            # //////////////////////////////////////////////////////////////////////////////////////
            l += 1

            if r==y_bar:
                #print('- A counterfactual is found!')
                d = self._edit_distance(g_o,g_c)
                return d,g_c,l

        return 0,g_o,l


    def get_prob_edges(self, gc, edges, y_bar, ki, edges_prob, p_0=0.5):
        '''
        '''
        gci = np.copy(gc)
        edges_prob_rem = np.array([])
        edges_prob_add = np.array([])
        edges_add = []
        edges_rem = []
        e = []

        # Modified code, unused variable remmoved //////////////////////////////////////////////
        # gd = g - gci
        # //////////////////////////////////////////////////////////////////////////////////////

        for e in edges:
            i,j = e
            if gc[i][j]>0:
                edges_prob_rem = np.append(edges_prob_rem,edges_prob[1-y_bar][i][j])
                edges_rem.append((i,j))
            else:
                edges_prob_add = np.append(edges_prob_add,edges_prob[y_bar][i][j])
                edges_add.append((i,j))
        edges_prob_add = edges_prob_add/edges_prob_add.sum()
        edges_prob_rem = edges_prob_rem/edges_prob_rem.sum()
        #print('-- ',len(edges_rem),len(edges_add),len(edges))
        edges_i = []
        kii=0
        while(kii<ki):
            kii+=1
            if self._bernoulli(p_0) and len(edges_add)>0:
                #add
                n = np.random.choice(range(len(edges_add)), size=1, p=edges_prob_add)[0]
                i,j = edges_add[n]
                gci[i][j]=1
                gci[j][i]=1
            elif len(edges_rem)>0:
                #remove
                n = np.random.choice(range(len(edges_rem)), size=1, p=edges_prob_rem)[0]
                i,j = edges_rem[n]
                gci[i][j]=0
                gci[j][i]=0
            edges_i.append((i,j))
        new_edges = edges_add+edges_rem
        return gci,new_edges


    def bb_prob_2(self, oracle: Oracle, g , gc1, y_bar, edges_prob, k=5, l_max=2000):
        '''
        '''
        info = []
        gc = np.copy(gc1)
        edges = self._get_change_list(g, gc)
        d = self._edit_distance(g,gc)
        li=0

        while(li<l_max and len(edges)>0 and d>1):
            ki = min(k,len(edges))
            #gci,edges,edges_i = get_prob_edges(g,edges,y_bar,k,edges_prob)
            gci,new_edges = self.get_prob_edges(gc,edges,y_bar,k,edges_prob)

            # Modified code ////////////////////////////////////////////////
            # r = oracle(gci)
            inst = DataInstance(-1)
            inst.from_numpy_array(gci)
            r = oracle.predict(inst)
            # //////////////////////////////////////////////////////////////
            li += 1

            if r==y_bar and self._edit_distance(gci,gc)>0:
                gc = np.copy(gci)
                d = self._edit_distance(g,gc)
                edges = self._get_change_list(g,gc)
                #print('ok --> ',r,d,li,k)
                info.append((r,d,li,ki))
                k+=1
            else:
                #print('no --> ',r,d,li,k)
                d = self._edit_distance(g,gc)
                info.append((r,d,li,ki))
                if k>1:
                    k-=1
                else:
                    edges.remove(new_edges[0])

        return gc, self._edit_distance(g, gc), li, info