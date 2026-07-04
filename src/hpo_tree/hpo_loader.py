"""
This module provides functions to download and load the Human Phenotype Ontology (HPO).
"""

import glob
import os.path
import warnings

import requests
from loguru import logger
from pronto import Ontology

warnings.filterwarnings("ignore")


def get_most_recent_hpo_file(data_dir: str):
    """
    Finds the most recently created .obo file in the specified directory.

    Args:
        data_dir (str): The directory to search for .obo files.

    Returns:
        str: The path to the most recent .obo file.

    Raises:
        Exception: If no .obo files are found in the specified directory.
    """
    hpo_files = glob.glob(os.path.join(data_dir, '*.obo'))
    if hpo_files:
        hpo_file = max(hpo_files, key=os.path.getctime)
    else:
        logger.error(
            f"No Human Phenotype Ontology found at {data_dir}. EXIT!")
        raise Exception()
    logger.debug(f'Loading existing *.obo file: {hpo_file}')
    return hpo_file


def load_hpo(output_dir: str, download: bool = True):
    """
    Loads the Human Phenotype Ontology, optionally downloading the latest version.

    If download is True, it attempts to download the latest hp.obo from the HPO GitHub repository.
    If the download fails or download is False, it falls back to the most recent local .obo file.

    Args:
        output_dir (str): The directory where the ontology file is saved or searched for.
        download (bool, optional): Whether to download the latest ontology file. Defaults to True.

    Returns:
        pronto.Ontology: The loaded HPO ontology object.
    """
    if download:
        # URL for latest HPO ontology file
        url = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo"

        try:
            logger.debug('Downloading Human Phenotype Ontology...')
            # Download and save hpo.obo
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes

            # Extract date from data-version line using regex
            date_match = response.text.split('\n')[1][-len('yyyy-mm-dd'):]

            # Save to current directory
            hpo_file = os.path.join(output_dir, f'hp_{date_match}.obo')
            with open(hpo_file, 'wb') as f:
                f.write(response.content)

            logger.debug(
                f"Downloaded Human Phenotype Ontology ({len(response.content) / 1024 / 1024:.1f} MB) to {os.path.abspath(hpo_file)}")
        except Exception as e:
            logger.warning('GitHub was not reachable to download current Human Phenotype Ontology.')
            hpo_file = get_most_recent_hpo_file(output_dir)
    else:
        # For debugging a workaround to not always download the HPO database
        hpo_file = get_most_recent_hpo_file(output_dir)

    hpo = Ontology(hpo_file)
    return hpo
