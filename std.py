import os
import jsonpickle
import numpy as np

stats_folder = './output/ablation_done'

# create a list of file and sub directories 
# names in the given directory 
oracle_dataset_folders = os.listdir(stats_folder)
files_list = []
# Iterate over all the oracle_dataset folders
for odf_entry in oracle_dataset_folders:
    # Create full path
    odf_full_Path = os.path.join(stats_folder, odf_entry)

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

            files_list.append(result_file_full_path)


result = {}
for stat_file_uri in files_list:
    with open(stat_file_uri, 'r') as stat_file_reader:
        stat_dict = jsonpickle.decode(stat_file_reader.read())

        explainer_name = stat_dict['config']['explainer']['name']

        ged = stat_dict['Graph_Edit_Distance']
        ged_std = np.std(ged)

        result[explainer_name] = ged_std

for k,v in result.items():
    print(f'{k}: {v}')

