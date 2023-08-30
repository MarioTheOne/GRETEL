import jsonpickle


def compose(config_file):
   pass




def _get_snippet(snippet_path):
    # Read the config dictionary inside the config path
    with open(snippet_path, 'r') as config_reader:
        snippet = jsonpickle.decode(config_reader.read())        
    config_reader.close

    return snippet


test_json = {
    "compose_st" : "./stores.json",
    "compose_ds" : "./dataset.json",
    "datasets": [
        {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 28, "n_in_cycles": 5, "nodes_per_cycle": 3, "number_of_cycles": 4} },
        {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 28, "n_in_cycles": 5, "nodes_per_cycle": 4, "number_of_cycles": 3} },
        {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 28, "n_in_cycles": 5, "nodes_per_cycle": 4, "number_of_cycles": 3} }
    ]
    }

logs = {"name": "log_store_path", "address": "./output/logs/"}
stores = {"store_paths": [
            {"name": "dataset_store_path", "address": "./data/datasets/"},
            {"name": "embedder_store_path", "address": "./data/embedders/"},
            {"name": "oracle_store_path", "address": "./data/oracles/"},
            {"name": "explainer_store_path", "address": "./data/explainers/"},
            {"name": "output_store_path", "address": "./output/"},
            {"compose" : "./logfile.json"}
        ]}


in_conf = test_json
out_conf = {}
for item in in_conf:
    #keys = item.keys()
    
    if item.startswith('compose_'):
        print(in_conf[item])
    else:
        out_conf[item] = in_conf[item]

    #values = item.values()
    #print(values)

print(out_conf)