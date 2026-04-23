"""
Main CLI module for training and utilities.

This module exposes Python Fire commands to:
- train: orchestrate data loading, dataset preparation, and delegate to TreeTrainer.
- train_all: iterate over a list of model identifiers and call train for each.
- clean_up: remove output/version directories for a given model type.
- export_twopi_graph_json: build the ontology graph and export the twopi layout.

Notes:
- Designed to be invoked from the command line via Python Fire.
- Documentation-only changes; no runtime behavior is altered.
"""
import itertools
import json
import os
import pathlib
import shutil
from typing import Literal, List

import fire
import numpy as np
import pandas as pd
import torch.cuda
import tqdm
from lightning import seed_everything
from loguru import logger

from lib.datasets.base_dataset import BaseFaceMeshDataset
from lib.datasets.gmdb_dataset import GMDBFaceMeshDataset
from lib.datasets.gmdb_hpo_dataset import GMDBFaceMeshHPODataset
from lib.datasets.utkface_dataset import UTKFaceFaceMeshDataset
from lib.hpo_tree.hpo_model import HumanPhenotypeModel
from lib.hpo_tree.hpo_term import HumanPhenotypeTerm
from lib.utils.mediapipe_helper import extract_face_meshes


def train(data_dir: str, out_dir: str, gmdb_root_dir: str, utk_root_dir: str = None,
          face_region_selections_file: str = None, use_face_outline: bool = False,
          use_meta_data: List[str] = [], db_type: Literal[-1, 0, 1, 2, 3, 4] = -1, folds: int = 5,
          feature_importance_threshold: float = 0.01, dimensions: Literal[2, 3] = 3, soft_labels: float = 0.0,
          min_samples_required: int = 50, max_num_workers: int = 20, seed: int = 42, parallel: bool = True):
    logger.info(f"Parameters: {', '.join([f'{key}={value}' for key, value in locals().items()])}")
    logger.info('CUDA is available!' if torch.cuda.is_available() else 'CUDA is not available!')

    # Compatibility for Slurm
    if isinstance(use_face_outline, str):
        use_face_outline = True
        use_face_outline = bool(use_face_outline)
    if isinstance(use_meta_data, str) and use_meta_data in ('null', 'None', '', '[]'):
        use_meta_data = []
    elif isinstance(use_meta_data, str):
        use_meta_data = json.loads(use_meta_data)
    if isinstance(feature_importance_threshold, str):
        feature_importance_threshold = float(feature_importance_threshold)
    if isinstance(dimensions, str):
        dimensions = int(dimensions)
    if isinstance(soft_labels, str):
        soft_labels = float(soft_labels)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    version = f'db={db_type}_d={dimensions}_f={use_face_outline}_m=[{"+".join(use_meta_data)}]_t={feature_importance_threshold:.2f}_l={soft_labels:.2f}_s={seed}'
    # logger_name = os.path.join(log_dir, f"{{time}}_{version}.log")
    # logger.remove()
    # logger.add(logger_name, level="DEBUG", enqueue=True, backtrace=True, diagnose=True)

    seed_everything(seed, workers=True)

    # Relevant data files
    gmdb_version = pathlib.Path(gmdb_root_dir).parts[-1].replace(".", "")
    gmdb_data_file = os.path.join(data_dir, f'gmdb_data_{gmdb_version}.csv')
    gmdb_hpo_data_file = os.path.join(data_dir, f'gmdb_hpo_facemesh_data_{gmdb_version}.csv')
    utk_data_file = os.path.join(data_dir, 'utk_data.csv')
    reference_face_file = os.path.join(data_dir, 'reference_face.jpg')

    # Load reference mesh
    reference_face_mesh = extract_face_meshes([reference_face_file])
    reference_face_mesh = reference_face_mesh[1].reshape((-1, 3))[:, :dimensions]

    # Load and prepare HPO tree
    hpo = HumanPhenotypeTerm.load_ontology(data_dir, download=False)
    hp_abnorm_face = hpo.find_successor('HP:0000271')  # Abnormality of the face
    hp_abnorm_eye = hpo.find_successor('HP:0000478')  # Abnormality of the eye
    hp_abnorm_eyebrow = hpo.find_successor('HP:0000534')  # Abnormal eyebrow morphology
    hp_abnorm_face.add_successor(hp_abnorm_eye)  # Move eye to face
    hp_abnorm_face.add_successor(hp_abnorm_eyebrow)  # Move eyebrow to face
    hp_abnorm_face.define_as_root()
    logger.debug(f'Prepared HPO tree and reduced it to the root {hp_abnorm_face.id} with {len(hp_abnorm_face)} nodes.')

    root_model = HumanPhenotypeModel.create_from_hpo(hp_abnorm_face, out_dir, dimensions, use_meta_data, parallel,
                                                     version, max_num_workers, recursive=True)

    if face_region_selections_file:
        # Face Mesh outline
        face_mesh_outline = [10, 338, 332, 297, 284, 251, 389, 356, 323, 454, 361, 288, 397, 365, 379, 378, 400, 152,
                             377, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        # Load manually annotated face region masks
        face_region_selections_file = os.path.join(data_dir, face_region_selections_file)
        with open(face_region_selections_file) as f:
            selections = json.load(f)
            point_annotations = selections['annotations']
            point_masks = {annotation['id']: annotation['points'] for annotation in point_annotations}
            for id, points in point_masks.items():
                point_mask = np.zeros(478, dtype=bool) if len(points) > 0 else np.ones(478, dtype=bool)
                if len(points) > 0:
                    points = list(set(points + face_mesh_outline)) if use_face_outline else points
                    point_mask[points] = True
                point_masks[id] = point_mask
    else:
        # If no face region masking has been done, let the models learn for themselves
        point_masks = {root_model.id: np.ones(478, dtype=bool)}
    point_mask = np.ones(478, dtype=bool)

    db_present = GMDBFaceMeshDataset('', gmdb_root_dir, gmdb_data_file, reference_mesh=reference_face_mesh,
                                     dimensions=dimensions)
    if db_type > -1:
        db_present = GMDBFaceMeshHPODataset('', data_dir, gmdb_data_file, gmdb_hpo_data_file,
                                            reference_mesh=reference_face_mesh, dimensions=dimensions)
        db_present.update_by_type(db_type)
        db_present.create_hpo_table(root_model, present_feature='present_features_merged',
                                    absent_feature='absent_features_merged')
    else:
        db_present.create_hpo_table(root_model)

    if utk_root_dir:
        # Prepare healthy faces dataset in case needed
        db_absent = UTKFaceFaceMeshDataset('age', utk_root_dir, utk_data_file,
                                           reference_mesh=reference_face_mesh, dimensions=dimensions)
        logger.debug('Using UTK Faces in the Wild for absent HPO features.')
    else:
        db_absent = db_present
        logger.debug('Using GMDB for absent HPO features.')

    with tqdm.tqdm(total=len(root_model), postfix=json.dumps({'Trained': 0, 'NoPoints': 0, 'NoSamples': 0})) as pbar:
        recursive_model_training(pbar, root_model, db_present, db_absent, point_mask, folds, seed,
                                 feature_importance_threshold, soft_labels, point_masks, min_samples_required)


# @logger.catch
def recursive_model_training(pbar: tqdm.tqdm, model: HumanPhenotypeModel, db_present: BaseFaceMeshDataset,
                             db_absent: BaseFaceMeshDataset, point_mask: np.ndarray, folds: int, seed: int,
                             feature_importance_threshold: float, soft_labels: float, available_point_masks,
                             min_samples_required: int):
    pbar.set_description_str(model.id)
    os.makedirs(model.log_path, exist_ok=True)

    if not os.path.exists(model.point_mask_file_output):
        # Check Pre-Conditions
        children_count = len(model)

        # In case a pre-defined point mask exists for this HPO-term
        if model.id in available_point_masks:
            point_mask = available_point_masks[model.id]
            logger.debug(f'{model.id} has a pre-defined point mask.')

        if point_mask is not None and point_mask.sum() < 2:
            logger.warning(f'Skip {model.id} ({model.name}) => {point_mask.sum()} points in the face mask.')
            update_pbar('NoPoints', 1 + children_count, pbar)
            return

        logger.debug(f'Using a point mask with {point_mask.sum()} points.')

        # Prepare Database
        logger.debug(f'Prepare dataset for node {model.id}...')
        dist_columns = ['gender', 'age', 'ethnicity']
        db_present.label_column = model.id
        # Prepare positive samples where HPO is present
        db_present_samples = db_present.sample_from_label_column(-1, 1, seed)
        present_sample_count = len(db_present_samples)
        if present_sample_count < folds:
            logger.warning(f'Skip {model.id} ({model.name}) => present samples: {present_sample_count}')
            update_pbar('NoSamples', 1 + children_count, pbar)
            shutil.rmtree(model.log_path)
            return
        if present_sample_count < min_samples_required // 2:
            logger.warning(f'Skip {model.id} ({model.name}) => present sample count {present_sample_count} too few')
            update_pbar('NoSamples', 1 + children_count, pbar)
            shutil.rmtree(model.log_path)
            return
        syndrome_counts = db_present_samples.data_df['internal_syndrome_name'].value_counts(dropna=False)
        syndrome_counts.index = syndrome_counts.index.where(syndrome_counts.index.notna(), other='Missing')
        syndrome_counts.to_csv(model.dist_syndrome_file)
        dict_distributions = db_present_samples.extract_distributions(columns=dist_columns)
        df_distributions_present = pd.concat(
            [pd.DataFrame(series) for series in dict_distributions.values()]).transpose()
        df_distributions_present['GMDB'] = present_sample_count
        df_distributions_present['UTK'] = 0
        df_distributions_present.to_csv(model.dist_hpo_present_file, index=False)
        # Prepare negative samples where HPO is absent
        db_absent_samples = db_present.sample_from_label_column(-1, 0, seed)
        half_of_present_samples_count = present_sample_count // 2  # Use 0-50% of GMDB negative samples if available
        if len(db_absent_samples) > half_of_present_samples_count:
            db_absent_samples = db_absent_samples.sample(half_of_present_samples_count, random_state=seed)
        negative_gmdb_sample_count = len(db_absent_samples)
        negative_utk_sample_count = present_sample_count - negative_gmdb_sample_count
        db_absent_samples = db_absent_samples.concat(
            db_absent.sample_by_distribution(present_sample_count - len(db_absent_samples), seed, dict_distributions))
        logger.debug(f'Using {negative_utk_sample_count} UTK samples and {negative_gmdb_sample_count} GMDB samples.')
        absent_sample_count = len(db_absent_samples)
        if absent_sample_count < folds:
            logger.warning(f'Skip {model.id} ({model.name}) => absent samples: {absent_sample_count}')
            update_pbar('NoSamples', 1 + children_count, pbar)
            shutil.rmtree(model.log_path)
            return
        # db_absent_samples = db_absent_samples.sample_by_distribution(len(db_present_samples), seed, dict_distributions)
        dict_distributions_absent = db_absent_samples.extract_distributions(columns=dist_columns)
        df_distributions_healthy = pd.concat(
            [pd.DataFrame(series) for series in dict_distributions_absent.values()]).transpose()
        df_distributions_healthy['GMDB'] = negative_gmdb_sample_count
        df_distributions_healthy['UTK'] = negative_utk_sample_count
        df_distributions_healthy.to_csv(model.dist_hpo_absent_file, index=False)
        # Combine positive + negative sampled datasets
        balanced_dataset = db_present_samples.concat(db_absent_samples, soft_labels)
        balanced_dataset.set_point_mask(point_mask)

        # Train HPO Model
        point_mask = model.train(balanced_dataset, folds, feature_importance_threshold, seed, point_mask)
    else:
        point_mask = np.load(model.point_mask_file_output)
    update_pbar('Trained', 1, pbar)
    for child_model in model.successors:
        # Continue with next HPO Models
        recursive_model_training(pbar, child_model, db_present, db_absent, point_mask, folds, seed,
                                 feature_importance_threshold, soft_labels, available_point_masks, min_samples_required)


def update_pbar(tag: Literal['Trained', 'NoPoints', 'NoSamples'], value: int, pbar: tqdm.tqdm):
    postfix = json.loads(pbar.postfix)
    postfix[tag] = postfix[tag] + value
    pbar.set_postfix_str(json.dumps(postfix))
    pbar.update(value)


if __name__ == '__main__':
    fire.Fire(train)
