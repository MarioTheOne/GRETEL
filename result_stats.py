from src.data_analysis.data_analyzer import DataAnalyzer

# data_store_path = './data/datasets/'
# dtan = DataAnalyzer('./output', './stats')

data_store_path = './data/datasets'
dtan = DataAnalyzer('/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/output/ablation_number_cycles', '/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/stats/ablation_number_cycles_full')
# dtan = DataAnalyzer('C:\\Work\\GNN\\Mine\\GRETEL2\\GRETEL\\output_legacy\\synthetic', 'C:\\Work\\GNN\\Mine\\GRETEL2\\GRETEL\\stats_legacy')

dtan.aggregate_data()
dtan.aggregate_runs()
dtan.create_tables_by_oracle_dataset()

# datasets =[
#     # {"name": "adhd", "parameters": {} },
#     {"name": "autism", "parameters": {} },
#     {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 28, "n_in_cycles": 7} },
#     # {"name": "tree-infinity", "parameters": {"n_inst": 500, "n_per_inst": 300, "n_infinities": 10, "n_broken_infinities": 10}},
#     # {"name": "bbbp", "parameters": {"force_fixed_nodes": False}},
#     # {"name": "hiv", "parameters": {"force_fixed_nodes": False}}
# ]

# dtan.get_datasets_stats(datasets, data_store_path)