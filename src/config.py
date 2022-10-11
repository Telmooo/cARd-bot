import yaml

from utils.utils import flatten_dictionary

DEBUG_MODE = False

def parse_config(yaml_file: str):

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    config = flatten_dictionary(config)

    return config