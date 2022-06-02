from src.dataset.dataset_factory import DatasetFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.explainer.explainer_factory import ExplainerFactory
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.evaluation.evaluator_base import Evaluator

import os
import jsonpickle

class EvaluatorManager:

    def __init__(self, config_file_path, run_number=0) -> None:
        
        # Check that the path to the config file exists
        if not os.path.exists(config_file_path):
            raise ValueError('''The provided config file does not exist''')

        # Read the config dictionary inside the config path
        with open(config_file_path, 'r') as config_reader:
            self._config_dict = jsonpickle.decode(config_reader.read())

        self._dataset_factory = None
        self._embedder_factory = None
        self._oracle_factory = None
        self._explainer_factory = None
        self._output_store_path = None
        self._evaluation_metric_factory = None
        self._run_number = run_number

        # iterate over the store paths and initialize the factories
        for store_path in self._config_dict['store_paths']:

            if(store_path['name'] == 'dataset_store_path'):
                dataset_store_path = store_path['address']

                # If the store folder does not exist then create it
                if not os.path.exists(dataset_store_path):
                    os.mkdir(dataset_store_path)

                # Create the factory with the corresponding store path
                self._dataset_factory = DatasetFactory(dataset_store_path)

            if(store_path['name'] == 'embedder_store_path'):
                embedder_store_path = store_path['address']
                
                # If the store folder does not exist then create it
                if not os.path.exists(embedder_store_path):
                    os.mkdir(embedder_store_path)

                # Create the factory with the corresponding store path
                self._embedder_factory = EmbedderFactory(embedder_store_path)

            if(store_path['name'] == 'oracle_store_path'):
                oracle_store_path = store_path['address']
                
                # If the store folder does not exist then create it
                if not os.path.exists(oracle_store_path):
                    os.mkdir(oracle_store_path)

                # Create the factory with the corresponding store path
                self._oracle_factory = OracleFactory(oracle_store_path)

            if(store_path['name'] == 'explainer_store_path'):
                explainer_store_path = store_path['address']
                
                # If the store folder does not exist then create it
                if not os.path.exists(explainer_store_path):
                    os.mkdir(explainer_store_path)

                # Create the factory with the corresponding store path
                self._explainer_factory = ExplainerFactory(explainer_store_path)

            if(store_path['name'] == 'output_store_path'):
                output_store_path = store_path['address']
                
                # If the store folder does not exist then create it
                if not os.path.exists(output_store_path):
                    os.mkdir(output_store_path)

                # store the output path
                self._output_store_path = output_store_path

        # Check that all the required store paths were provided
        if ((self._dataset_factory is None) or (self._embedder_factory) is None or 
            (self._oracle_factory is None) or (self._explainer_factory is None) or
            (self._output_store_path is None)):
            raise ValueError('''Not all required store paths were provided''')

        # Create the evaluation metrics factory
        self._evaluation_metric_factory = EvaluationMetricFactory()

        self.datasets = []
        self.oracles = []
        self.explainers = []
        self.evaluators = []
        self.evaluation_metrics = []

    
    def generate_synthetic_datasets(self):
        """Generates the synthetic datasets and stores them on disk, allowing them to be loaded later
         by the other methods without the need for them to be generated on the fly

        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        # Given the datasets are generated on the fly, this method only calls the get_dataset_by_name method
        for dataset_dict in self._config_dict['datasets']:
            self._dataset_factory.get_dataset_by_name(dataset_dict)


    def train_oracles(self):
        """Trains the oracles and store them on disk, allowing to use them later without needing
         to train again. Trains one oracle of each kind for each dataset.
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        dataset_dicts = self._config_dict['datasets']
        oracle_dicts = self._config_dict['oracles']

        # Create the datasets needed for fitting the oracles
        for dataset_dict in dataset_dicts:
            self.datasets.append(self._dataset_factory.get_dataset_by_name(dataset_dict))

        # For each dataset create and fit all oracles
        for dataset in self.datasets:
            for oracle_dict in oracle_dicts:

                # The get_oracle_by_name method returns a fitted oracle
                oracle = self._oracle_factory.get_oracle_by_name(oracle_dict, dataset, self._embedder_factory)
                self.oracles.append(oracle)

        # The goal of this method is to train the oracles and store them on disk to use them later
        # For this reason we clean the datasets and oracles lists after using them
        self.datasets = []
        self.oracles = []

        
    def create_evaluators(self):
        """Creates one evaluator for each combination of dataset-oracle-explainer using the chosen metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        dataset_dicts = self._config_dict['datasets']
        oracle_dicts = self._config_dict['oracles']
        metric_dicts = self._config_dict['evaluation_metrics']
        explainer_dicts = self._config_dict['explainers']

        # Create the datasets
        for dataset_dict in dataset_dicts:
            self.datasets.append(self._dataset_factory.get_dataset_by_name(dataset_dict))

        # Create the evaluation metrics
        for metric_dict in metric_dicts:
            eval_metric = self._evaluation_metric_factory.get_evaluation_metric_by_name(metric_dict)
            self.evaluation_metrics.append(eval_metric)

        for explainer_dict in explainer_dicts:
            explainer = self._explainer_factory.get_explainer_by_name(explainer_dict, self._evaluation_metric_factory)
            self.explainers.append(explainer)

        evaluator_id = 0
        for dataset in self.datasets:
            for explainer in self.explainers:
                for oracle_dict in oracle_dicts:

                    # The get_oracle_by_name method returns a fitted oracle
                    oracle = self._oracle_factory.get_oracle_by_name(oracle_dict, dataset, self._embedder_factory)

                    # Creating the evaluator
                    evaluator = Evaluator(evaluator_id, dataset, oracle, explainer, self.evaluation_metrics,
                                             self._output_store_path, self._run_number)

                    # Adding the evaluator to the evaluator's list
                    self.evaluators.append(evaluator)

                    # increasing the evaluator id counter
                    evaluator_id +=1

                
    def evaluate(self):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        for evaluator in self.evaluators:
            evaluator.evaluate()

        
    def evaluate_multiple_runs(self, n_runs):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics.
        Each evaluator is run "n_runs" times
         
        -------------
        INPUT: n_runs the number of times the evaluate method of each evaluator is going to be called

        -------------
        OUTPUT: None
        """
        for evaluator in self.evaluators:
            for i in range(0, n_runs):
                evaluator.evaluate()