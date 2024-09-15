
from types import SimpleNamespace

import argparse
import os 
import yaml

CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'beam_config.yaml'
)


# Function to load defaults from config file
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args(config):
    """Define and parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description='Load config and override with command-line arguments.')

    # Define your command-line arguments here
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value, required=False)

    return parser.parse_known_args()


def update_config(config, args):
    """Update the config dictionary with command-line arguments."""
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def dict_to_namespace(d):
    """
    Recursively convert a dictionary to a SimpleNamespace for dot notation access.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)



def load_pipeline_config(config_file = CONFIG_FILE_PATH):
    """
    Load config file and override with command-line arguments.

    NOTE: If conflicting parameter names, both cases 
    will be updated with value from command-line arg.
    """
    config = load_config(config_file)["pipeline"]
    for stage in config.keys():
        args, _ = parse_args(config[stage])
        config[stage] = update_config(config[stage], args)
    return dict_to_namespace(config)


if __name__ == "__main__":

    print(load_pipeline_config())
