from matplotlib.pyplot import table
from src.dataset.dataset_factory import DatasetFactory

import os
import jsonpickle
import numpy as np
from sklearn import metrics
import pandas as pd


class DataAnalyzer():

    def __init__(self, stats_folder, output_folder) -> None:
        self.stats_folder = stats_folder
        self.output_folder = output_folder
        self.data_dict = {}

    
    def aggregate_data(self):
        stat_files = self._get_files_list()

        for stat_file_uri in stat_files:
            with open(stat_file_uri, 'r') as stat_file_reader:
                stat_dict = jsonpickle.decode(stat_file_reader.read())

                dataset_name = stat_dict['config']['dataset']['name']
                oracle_name = stat_dict['config']['oracle']['name']
                explainer_name = stat_dict['config']['explainer']['name']

                if dataset_name not in self.data_dict:
                    self.data_dict[dataset_name] = { oracle_name: {explainer_name: [] }}

                elif oracle_name not in self.data_dict[dataset_name]:
                    self.data_dict[dataset_name][oracle_name] = { explainer_name: [] }

                elif explainer_name not in self.data_dict[dataset_name][oracle_name]:
                    self.data_dict[dataset_name][oracle_name][explainer_name] = []

                # getting the data for each metric
                metrics = {}
                for k, v in stat_dict.items():
                    if k != 'config':
                        v_mean = np.mean(v)
                        metrics[k] = v_mean

                self.data_dict[dataset_name][oracle_name][explainer_name].append(metrics)


    def aggregate_runs(self):
        # for each dataset
        for dataset in self.data_dict:
            # for each oracle
            for oracle in self.data_dict[dataset]:
                # for each explainer
                for explainer, runs in self.data_dict[dataset][oracle].items():
                    # for each run
                    metrics = {}
                    for run in runs:
                        for metric, value in run.items():
                            if metric not in metrics:
                                metrics[metric] = []
                            metrics[metric].append(value)

                    aggregated_metrics = {}
                    for metric, values in metrics.items():
                        aggregated_metrics[metric] = np.mean(values)
                        aggregated_metrics[metric + '-std'] = np.std(values)

                    self.data_dict[dataset][oracle][explainer] = aggregated_metrics


    def create_tables_by_oracle_dataset(self):
        # iterate over all datasets
        for dataset in self.data_dict:
            # iterate over all oracles
            for oracle in self.data_dict[dataset]:
                # For each dataset-oracle create a table
                data = {'explainer': []}
                for explainer in self.data_dict[dataset][oracle]:
                    data['explainer'].append(explainer)
                    for metric, value in self.data_dict[dataset][oracle][explainer].items():
                        if not metric in data:
                            data[metric] = []
                        data[metric].append(value)

                table = pd.DataFrame(data)
                
                table_path_csv = os.path.join(self.output_folder, dataset + '-' + oracle + '.csv')
                table_path_tex = os.path.join(self.output_folder, dataset + '-' + oracle + '.tex')

                table.to_csv(table_path_csv)
                table.to_latex(table_path_tex)


    def get_datasets_stats(self, dataset_dicts, data_store_path):
        d_fact = DatasetFactory(data_store_path)
        datasets = []

        for d_dict in dataset_dicts:
            ds = d_fact.get_dataset_by_name(d_dict)
            datasets.append(ds)

        data = {'name': [], 'instance_number': [], 'nodes_mean': [], 'nodes_std':[], 'edges_mean':[], 'edges_std': [], 'inst_class_0':[], 'inst_class_1':[]}

        for dts in datasets:
            nodes = np.array([len(i.graph.nodes) for i in dts.instances])
            edges = np.array([len(i.graph.edges) for i in dts.instances])

            class_0_count = 0
            class_1_count = 0
            for i in dts.instances:
                if i.graph_label == 0:
                    class_0_count +=1
                else:
                    class_1_count +=1

            data['name'].append(dts.name)
            data['instance_number'].append(len(dts.instances))
            data['nodes_mean'].append(np.mean(nodes))
            data['nodes_std'].append(np.std(nodes))
            data['edges_mean'].append(np.mean(edges))
            data['edges_std'].append(np.std(edges))
            data['inst_class_0'].append(class_0_count)
            data['inst_class_1'].append(class_1_count)

            table = pd.DataFrame(data)
            table_path_csv = os.path.join(self.output_folder, 'dataset_stats.csv')
            table_path_tex = os.path.join(self.output_folder, 'dataset_stats.tex')

            table.to_csv(table_path_csv)
            table.to_latex(table_path_tex)


    def _get_files_list(self):
        # create a list of file and sub directories 
        # names in the given directory 
        oracle_dataset_folders = os.listdir(self.stats_folder)
        result = []
        # Iterate over all the oracle_dataset folders
        for odf_entry in oracle_dataset_folders:
            # Create full path
            odf_full_Path = os.path.join(self.stats_folder, odf_entry)

            # Get the explainer folders for that oracle-dataset combo
            explainer_folders_list = os.listdir(odf_full_Path)

            # Iterate over all explainer folders
            for exf_entry in explainer_folders_list:
                # Get the explainer folders full path
                exf_full_path = os.path.join(odf_full_Path, exf_entry)

                # get all the results files for that explainer-oracle-dataset combo
                result_files = os.listdir(exf_full_path)

                # iterate over the result files
                for result_entry in result_files:
                    result_file_full_path = os.path.join(exf_full_path, result_entry)

                    result.append(result_file_full_path)
            
                    
        return result


