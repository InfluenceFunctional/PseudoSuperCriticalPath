from argparse import Namespace
from pathlib import Path

import yaml


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def load_yaml(path):
    """
    Safely load yaml file as dict.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


def process_config(cmdline_args):
    cmdline_keys = [
        arg.strip("--").split("=")[0] for arg in cmdline_args if "--" in arg
    ]
    cmdline_values = [
        arg for arg in cmdline_args if "--" not in arg
    ]
    cmdline_dict = {key: val for key, val in zip(cmdline_keys, cmdline_values)}
    config = dict2namespace(load_yaml(cmdline_dict['yaml_config']))
    return config