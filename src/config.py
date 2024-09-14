from types import SimpleNamespace
import yaml
import os


def dict_to_namespace(d):
    """
    Recursively convert a dictionary to a SimpleNamespace for dot notation access.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def load(section="stages", config_path=None):
    """
    Load configuration from a YAML file and convert the section to a SimpleNamespace.
    """
    if config_path is None:
        # Get the absolute path of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the default config path relative to this script's directory
        config_path = os.path.join(current_dir, 'config.yaml')

    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    print(f"Loaded {config_path} - section: {section}")
    
    # Convert the section to a SimpleNamespace for dot notation
    return dict_to_namespace(config_data[section])