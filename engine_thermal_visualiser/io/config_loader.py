import yaml

def load_tc_config(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data
