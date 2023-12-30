# função para utilizar o config.yaml

import os
import yaml


def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.join('..', '..', 'config', 'config.yaml')

    config_file_path = os.path.abspath(os.path.join(current_dir, rel_path))

    config_file = yaml.safe_load(open(config_file_path, 'rb'))
    return config_file

load_config()