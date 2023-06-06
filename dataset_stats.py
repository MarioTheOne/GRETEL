from src.data_analysis.data_analyzer import DataAnalyzer

# data_store_path = './data/datasets/'
# dtan = DataAnalyzer('./output', './stats')

data_store_path = './data/datasets'
dtan = DataAnalyzer('./output/steel/BBBP', './stats/steel/')


datasets =[
    {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 32, "n_in_cycles": 7} }
]

dtan.get_datasets_stats(datasets, data_store_path)