import os
import sys
import yaml
from pathlib import Path
from box import ConfigBox
from heartpipeline.logging import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read yaml file and return ConfigBox object
    
    Args:
        path_to_yaml (Path): Path to yaml file
        
    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories
    
    Args:
        path_to_directories (list): List of paths to directories
        verbose (bool, optional): Ignore if multiple directories. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")
