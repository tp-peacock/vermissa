import yaml
import os

def load_config(config_path):
    with open(config_path, "r") as f:
        config =  yaml.safe_load(f)
    config['data']['root'] = os.path.expandvars(config['data']['root'])
    config['model']['output_dir'] = os.path.expandvars(config['model']['output_dir'])
    config['model']['training']['logging_dir'] = os.path.expandvars(config['model']['training']['logging_dir'])

    return config