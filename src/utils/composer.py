import jsonpickle

def compose(config):
    out_conf  = {}
    for item in config:
        if item.startswith('compose'):
            snippet = _get_snippet(config[item])
            for sub in snippet:
                out_conf[sub]=_process_array(snippet[sub])#compose(snippet[sub])
        else:
            out_conf[item] = _process_array(config[item])

    return out_conf

def _process_array(conf):
    if (isinstance(conf, list)):
        out_arr = []
        for arr_item in conf:
            out_arr.append(compose(arr_item))
        return out_arr
    return conf

def _get_snippet(snippet_path):
    # Read the config dictionary inside the config path
    with open(snippet_path, 'r') as config_reader:
        snippet = jsonpickle.decode(config_reader.read())        
    config_reader.close()

    return snippet



'''snippet_path = 'config/test/test_compose.json'
#snippet_path = 'config/test/comp_stores.json'
with open(snippet_path, 'r') as config_reader:
    in_conf = jsonpickle.decode(config_reader.read())        

out_conf = compose(in_conf)

print(out_conf)'''