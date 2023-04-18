from src.explainer.explainer_perturbation_rand import PerturbationRandExplainer
from src.explainer.meg.explainer_meg import MEGExplainer
from src.explainer.meg.utils.encoders import MorganBitFingerprintActionEncoder, IDActionEncoder
from src.explainer.meg.environments.basic_policies import AddRemoveEdgesEnvironment
from src.dataset.converters.weights_converter import DefaultFeatureAndWeightConverter
from src.dataset.converters.cf2_converter import CF2TreeCycleConverter
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.explainer.ensemble.ensemble_factory import EnsembleFactory
from src.explainer.explainer_base import Explainer
from src.explainer.explainer_bidirectional_search import (
    DataDrivenBidirectionalSearchExplainer,
    ObliviousBidirectionalSearchExplainer)
from src.explainer.explainer_cf2 import CF2Explainer
from src.explainer.explainer_cfgnnexplainer import CFGNNExplainer
from src.explainer.explainer_clear import CLEARExplainer
from src.explainer.explainer_countergan import CounteRGANExplainer
from src.explainer.explainer_dce_search import (DCESearchExplainer,
                                                DCESearchExplainerOracleless)
from src.explainer.explainer_maccs import MACCSExplainer


class ExplainerFactory:

    def __init__(self, explainer_store_path) -> None:
        self._explainer_id_counter = 0
        self._explainer_store_path = explainer_store_path
        self._ensemble_factory = EnsembleFactory(explainer_store_path, self)

    def get_explainer_by_name(self, explainer_dict, metric_factory : EvaluationMetricFactory) -> Explainer:
        explainer_name = explainer_dict['name']
        explainer_parameters = explainer_dict['parameters']

        # Check if the explainer is DCE Search
        if explainer_name == 'dce_search':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''DCE Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])

            # Returning the explainer
            return self.get_dce_search_explainer(dist_metric, explainer_dict)

        # Check if the explainer is DCE Search Oracleless
        elif explainer_name == 'dce_search_oracleless':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''DCE Search Oracleless requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_dce_search_explainer_oracleless(dist_metric, explainer_dict)

        elif explainer_name == 'bidirectional_oblivious_search':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''Bidirectional Oblivious Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_bidirectional_oblivious_search_explainer(dist_metric, explainer_dict)

        elif explainer_name == 'bidirectional_data-driven_search':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''Bidirectional Data-Driven Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_bidirectional_data_driven_search_explainer(dist_metric, explainer_dict)

        elif explainer_name == 'maccs':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''Bidirectional Data-Driven Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_maccs_explainer(dist_metric, explainer_dict)

        
        elif explainer_name == 'cfgnnexplainer':
            # Returning the explainer
            return self.get_cfgnn_explainer(explainer_dict)

        
        elif explainer_name == 'ensemble':
            # Returning the ensemble explainer
            return self.get_ensemble(explainer_dict, metric_factory)
        
        elif explainer_name == 'countergan':
            # Verifying the explainer parameters
            if not 'n_nodes' in explainer_parameters:
                raise ValueError('''CounteRGAN requires the number of nodes''')
            if not 'device' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a device''')
            if not 'n_labels' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a n_labels''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a fold_id''')
            
            n_nodes = int(explainer_parameters['n_nodes'])
            batch_size_ratio = explainer_parameters.get('batch_size_ratio', .1)
            device = explainer_parameters['device']
            training_iterations = explainer_parameters.get('training_iterations', 20000)
            n_generator_steps = explainer_parameters.get('n_generator_steps', 2)
            n_discriminator_steps = explainer_parameters.get('n_discriminator_steps', 3)
            n_labels = explainer_parameters['n_labels']
            fold_id = explainer_parameters['fold_id']
            ce_binarization_threshold = explainer_parameters.get('ce_binarization_threshold', None)
            
            return self.get_countergan_explainer(n_nodes, batch_size_ratio, device,
                                                 training_iterations, n_discriminator_steps,
                                                 n_generator_steps, n_labels, fold_id,
                                                 ce_binarization_threshold, explainer_dict)
            
        elif explainer_name == 'clear':
            # Verifying the explainer parameters
            if not 'n_nodes' in explainer_parameters:
                raise ValueError('''CLEAR requires the number of nodes''')
            if not 'n_labels' in explainer_parameters:
                raise ValueError('''CLEAR requires a n_labels''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''CLEAR requires a fold_id''')
        
            batch_size_ratio = explainer_parameters.get('batch_size_ratio', .1)
            h_dim = explainer_parameters.get('h_dim', 16)
            z_dim = explainer_parameters.get('z_dim', 16)
            dropout = explainer_parameters.get('dropout', .1)
            encoder_type = explainer_parameters.get('encoder_type', 'gcn')
            disable_u = explainer_parameters.get('disable_u', False)
            lr = explainer_parameters.get('lr', 1e-3)
            weight_decay = explainer_parameters.get('weight_decay', 1e-5)
            graph_pool_type = explainer_parameters.get('graph_pool_type', 'mean')
            epochs = explainer_parameters.get('epochs', 200)
            alpha = explainer_parameters.get('alpha', 5)
            beta_x = explainer_parameters.get('beta_x', 10)
            beta_adj = explainer_parameters.get('beta_adj', 10)
            feature_dim = explainer_parameters.get('feature_dim', 2)
            lambda_sim = explainer_parameters.get('lambda_sim', 1)
            lambda_kl = explainer_parameters.get('lambda_kl', 1)
            lambda_cfe = explainer_parameters.get('lambda_cfe', 1)
            
            assert feature_dim >= 2
            
            n_nodes = int(explainer_parameters['n_nodes'])
            n_labels = int(explainer_parameters['n_labels'])
            fold_id = int(explainer_parameters['fold_id'])
    
            # max_num_nodes here is equal to n_labels
            # the authors use it originally to pad the graph adjacency matrices
            # if they're different within the dataset instances.
            return self.get_clear_explainer(n_nodes, n_labels, batch_size_ratio,
                                            h_dim, z_dim, dropout,
                                            encoder_type, graph_pool_type, disable_u,
                                            epochs, alpha, beta_x, beta_adj,
                                            feature_dim, lr, weight_decay,
                                            lambda_sim, lambda_kl, lambda_cfe,
                                            fold_id, explainer_dict)
            
        elif explainer_name == 'cf2':
            if not 'n_nodes' in explainer_parameters:
                raise ValueError('''CF2 requires the number of nodes''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''CF2 requires a fold_id''')
            
            batch_size_ratio = explainer_parameters.get('batch_size_ratio', .1)
            lr = explainer_parameters.get('lr', 1e-3)
            weight_decay = explainer_parameters.get('weight_decay', 1e-5)
            gamma = explainer_parameters.get('gamma', 'mean')
            epochs = explainer_parameters.get('epochs', 200)
            alpha = explainer_parameters.get('alpha', 5)
            lam = explainer_parameters.get('lambda', 5)
            feature_dim = explainer_parameters.get('feature_dim', 8)
            weight_dim = explainer_parameters.get('weight_dim', 3)
            converter_name = explainer_parameters.get('converter', 'tree_cycles')
            
            n_nodes = int(explainer_parameters['n_nodes'])
            fold_id = int(explainer_parameters['fold_id'])
            
            converter = None
            if converter_name == 'tree_cycles':
                converter = CF2TreeCycleConverter(feature_dim=feature_dim)
            else:
                converter = DefaultFeatureAndWeightConverter(feature_dim=feature_dim,
                                                              weight_dim=n_nodes)
                 
            return self.get_cff_explainer(n_nodes, converter,
                                          batch_size_ratio, lr, weight_decay,
                                          gamma, lam, alpha, epochs,
                                          fold_id, explainer_dict)
            
        elif explainer_name == 'meg':
            if not 'env' in explainer_parameters:
                raise ValueError('''MEG requires an environment to run reinforcement learning''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''MEG requires a fold_id''')
            if not 'action_encoder' in explainer_parameters:
                raise ValueError('''MEG requires an action_encoder''')
            
            environment = None
            env_name = explainer_parameters['env'].get('name', None)
            if not env_name:
                raise ValueError('''MEG requires to have and environment name''')
            else:
                if env_name in ['tree-cycles', 'asd']:
                    environment = AddRemoveEdgesEnvironment(*explainer_parameters['env']['args'])
                    """elif env_name == 'asd':
                    print(explainer_parameters['env']['args'])
                    environment = ASDEnvironment(**explainer_parameters['env']['args'])"""
                else:
                    raise ValueError('''MEG supports only "tree-cycles" for environment''')
                
            action_encoder = explainer_parameters['action_encoder'].get('name', 'tree-cycles')
            if action_encoder == 'id':
                action_encoder = IDActionEncoder()
            elif action_encoder == 'morgan_bit_fingerprint':
                action_encoder = MorganBitFingerprintActionEncoder(*explainer_parameters['action_encoder']['args'])
            else:
                raise ValueError('''MEG supports only "tree-cycles", "morgan_bit_fingerprint" to encode actions''')
            
            num_input = explainer_parameters.get('num_input', 5)    
            batch_size = explainer_parameters.get('batch_size', 1)
            lr = explainer_parameters.get('lr', 1e-4)
            replay_buffer_size = explainer_parameters.get('replay_buffer_size', 10000)
            num_epochs = explainer_parameters.get('epochs', 10)
            max_step_per_episode = explainer_parameters.get('max_steps_per_episode', 1)
            update_interval = explainer_parameters.get('update_interval', 1)
            gamma = explainer_parameters.get('gamma', 0.95)
            polyak = explainer_parameters.get('polyak', 0.995)
            sort_predicate = lambda result : result['reward']
            num_counterfactuals = explainer_parameters.get('num_counterfactuals', 10)
            
            
            fold_id = int(explainer_parameters['fold_id'])
            
            return self.get_meg_explainer(environment, action_encoder,
                                          num_input, replay_buffer_size,
                                          num_epochs, max_step_per_episode,
                                          update_interval, gamma, polyak,
                                          sort_predicate, fold_id, num_counterfactuals,
                                          batch_size,
                                          explainer_dict)
            
        elif explainer_name == 'perturbation_rand':
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''MEG requires a fold_id''')
            
            perturbation_percentage = explainer_parameters.get('perturbation_percentage', .05)
            fold_id = explainer_parameters['fold_id']
            
            return self.get_perturb_rand_explainer(fold_id, perturbation_percentage, explainer_dict)
        else:
            raise ValueError('''The provided explainer name does not match any explainer provided 
            by the factory''')

    def get_perturb_rand_explainer(self, fold_id, perturbation_percentage, config_dict=None):
        result = PerturbationRandExplainer(id=self._explainer_id_counter,
                                         perturbation_percentage=perturbation_percentage,
                                         fold_id=fold_id,
                                         config_dict=config_dict)
        self._explainer_id_counter += 1
        return result
        
        
    def get_meg_explainer(self, environment, action_encoder,
                          num_input,
                          replay_buffer_size, num_epochs,
                          max_step_per_episode, update_interval, gamma,
                          polyak, sort_predicate, fold_id, num_counterfactuals,
                          batch_size,
                          config_dict=None) -> Explainer:
        
        result = MEGExplainer(id=self._explainer_id_counter,
                              environment=environment,
                              num_counterfactuals=num_counterfactuals,
                              action_encoder=action_encoder,
                              batch_size=batch_size,
                              replay_buffer_size=replay_buffer_size,
                              num_epochs=num_epochs,
                              max_steps_per_episode=max_step_per_episode,
                              update_interval=update_interval,
                              gamma=gamma,
                              polyak=polyak,
                              sort_predicate=sort_predicate,
                              fold_id=fold_id,
                              num_input=num_input,
                              config_dict=config_dict)
        self._explainer_id_counter += 1
        return result

    def get_dce_search_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = DCESearchExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1  
        return result

    def get_dce_search_explainer_oracleless(self, instance_distance_function, config_dict=None) -> Explainer:
        result = DCESearchExplainerOracleless(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_bidirectional_oblivious_search_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = ObliviousBidirectionalSearchExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_bidirectional_data_driven_search_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = DataDrivenBidirectionalSearchExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_maccs_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = MACCSExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_cfgnn_explainer(self, config_dict=None) -> Explainer:
        result = CFGNNExplainer(self._explainer_id_counter, config_dict)
        self._explainer_id_counter += 1
        return result

    
    def get_ensemble(self, config_dic = None, metric_factory : EvaluationMetricFactory = None) -> Explainer:
        result = self._ensemble_factory.build_explainer(config_dic, metric_factory)
        self._explainer_id_counter += 1
        return result
    
    
    def get_cff_explainer(self, n_nodes, converter,
                          batch_size_ratio,
                          lr, weight_decay, gamma, lam,
                          alpha, epochs,
                          fold_id, config_dict=None) -> Explainer:
        result = CF2Explainer(self._explainer_id_counter,
                              self._explainer_store_path,
                              n_nodes=n_nodes,
                              batch_size_ratio=batch_size_ratio,
                              lr=lr,
                              weight_decay=weight_decay,
                              gamma=gamma,
                              lam=lam,
                              alpha=alpha,
                              epochs=epochs,
                              converter=converter,
                              fold_id=fold_id,
                              config_dict=config_dict)
        self._explainer_id_counter += 1
        return result
    
    def get_countergan_explainer(self, n_nodes, batch_size_ratio, device,
                                 training_iterations, n_discriminator_steps, n_generator_steps,
                                 n_labels, fold_id, ce_binarization_threshold, config_dict=None) -> Explainer:
        result = CounteRGANExplainer(self._explainer_id_counter,
                                     self._explainer_store_path,
                                     n_nodes=n_nodes,
                                     batch_size_ratio=batch_size_ratio,
                                     device=device,
                                     n_labels=n_labels,
                                     training_iterations=training_iterations,
                                     n_generator_steps=n_generator_steps,
                                     n_discriminator_steps=n_discriminator_steps,
                                     ce_binarization_threshold=ce_binarization_threshold,
                                     fold_id=fold_id, config_dict=config_dict)
        self._explainer_id_counter += 1
        return result
       
   
    def get_clear_explainer(self, n_nodes, n_labels, batch_size_ratio,
                            h_dim, z_dim, dropout, encoder_type, graph_pool_type,
                            disable_u, epochs, alpha, beta_x, beta_adj, feature_dim,
                            lr, weight_decay, lambda_sim, lambda_kl, lambda_cfe,
                            fold_id, config_dict=None) -> Explainer:
        
        result = CLEARExplainer(self._explainer_id_counter,
                                self._explainer_store_path,
                                n_nodes=n_nodes,
                                n_labels=n_labels,
                                batch_size_ratio=batch_size_ratio,
                                h_dim=h_dim,
                                z_dim=z_dim,
                                dropout=dropout,
                                encoder_type=encoder_type,
                                graph_pool_type=graph_pool_type,
                                disable_u=disable_u,
                                epochs=epochs,
                                alpha=alpha,
                                beta_x=beta_x,
                                beta_adj=beta_adj,
                                feature_dim=feature_dim,
                                lr=lr,
                                weight_decay=weight_decay,
                                lambda_sim=lambda_sim,
                                lambda_kl=lambda_kl,
                                lambda_cfe=lambda_cfe,
                                fold_id=fold_id,
                                config_dict=config_dict)
        self._explainer_id_counter += 1
        return result
        
        
        