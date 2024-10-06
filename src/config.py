
from types import SimpleNamespace

import apache_beam as beam
import argparse
import os 
import yaml

CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'config/common.yaml'
)
ENV = os.environ.get('ENV', "local")
if ENV == "local":
    EXTRA_CONFIG_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'config/local.yaml'
    )
elif ENV == "gcp":
    EXTRA_CONFIG_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'config/gcp.yaml'
    )
else:
    raise ValueError(f"Invalid ENV: {ENV}")


def add_write_params(config):
    config["bigquery"] = {
        "method": beam.io.WriteToBigQuery.Method.FILE_LOADS,
        "create_disposition": beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        "write_disposition": beam.io.BigQueryDisposition.WRITE_APPEND
    }
    return config


def read_config(config_file):
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


def append_values(config, extra_config):
    """
    Append values to the config dictionary.
    """
    for stage in config.keys():
        if stage in extra_config:
            config[stage].update(extra_config[stage])

    return config


def update_config(config):
    """
    Update the config dictionary with command-line arguments.
    
    NOTE: If conflicting parameter names, both cases 
    will be updated with value from command-line arg.
    """
    for stage in config.keys():
        args, _ = parse_args(config[stage])
        for key, value in vars(args).items():
            if value is not None:
                config[stage][key] = value
    return config


def dict_to_namespace(d):
    """
    Recursively convert a dictionary to a SimpleNamespace for dot notation access.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_pipeline_config(
        config_file = CONFIG_FILE_PATH,
        extra_config_file = EXTRA_CONFIG_FILE_PATH
    ):
    """
    Load config file and override with command-line arguments.

    NOTE: If conflicting parameter names, both cases 
    will be updated with value from command-line arg.
    """
    # params from config files
    config = read_config(config_file)["pipeline"]
    extra_config = read_config(extra_config_file)["pipeline"]
    config = append_values(config, extra_config)

    # add write parameters
    config = add_write_params(config)

    # command-line arguments
    config = update_config(config)

    return dict_to_namespace(config)


if __name__ == "__main__":

    print(load_pipeline_config())
