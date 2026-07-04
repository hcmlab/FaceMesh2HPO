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

from src.datasets.base_dataset import BaseFaceMeshDataset
from src.datasets.gmdb_dataset import GMDBFaceMeshDataset
from src.datasets.gmdb_hpo_dataset import GMDBFaceMeshHPODataset
from src.datasets.utkface_dataset import UTKFaceFaceMeshDataset
from src.hpo_tree.hpo_model import HumanPhenotypeModel
from src.utils.hpo_graph import build_modified_hpo_tree
from src.utils.mediapipe_helper import extract_face_meshes


def ablation_study(data_dir: str, out_dir: str, gmdb_root_dir: str, utk_root_dir: str = None,
                   face_region_selections_file: str = None, db_type: Literal[-1, 0, 1, 2, 3, 4] = -1, folds: int = 5,
                   seed: int = 42, parallel: bool = True):
    """Run a grid-based ablation study over key training settings.

    This function iterates over predefined combinations of feature dimensions,
    face outline usage, soft-label strengths, feature-importance thresholds,
    and metadata settings. For each configuration, it launches a full training run.

    Args:
        data_dir: Directory containing prepared input data files.
        out_dir: Directory where model outputs and logs are written.
        gmdb_root_dir: Root directory of the GMDB dataset.
        utk_root_dir: Optional root directory of the UTKFace dataset.
        face_region_selections_file: Optional JSON file containing manually defined face-region masks.
        db_type: Database subtype used to filter or specialize GMDB-HPO training data.
        folds: Number of cross-validation folds.
        seed: Random seed used for reproducibility.
        parallel: Whether to enable parallel model creation and processing.
    """
    dimensions = [3, 2]
    use_face_outlines = [False, True]
    soft_labels = [0, 0.05, 0.1]
    feature_importance_thresholds = [0.01, 0.05, 0.1]
    use_meta_data = [[], ['age', 'gender', 'ethnicity']]
    for config in itertools.product(dimensions, use_face_outlines, soft_labels, feature_importance_thresholds,
                                    use_meta_data):
        dimension, use_outline, soft_label, feature_importance_threshold, meta_data = config
        train(data_dir, out_dir, gmdb_root_dir, utk_root_dir, face_region_selections_file=face_region_selections_file,
              use_face_outline=use_outline, use_meta_data=meta_data, db_type=db_type, folds=folds,
              feature_importance_threshold=feature_importance_threshold, dimensions=dimension,
              soft_labels=soft_label, seed=seed, parallel=parallel)


def train(data_dir: str, out_dir: str, gmdb_root_dir: str, utk_root_dir: str = None,
          face_region_selections_file: str = None, use_face_outline: bool = False,
          use_meta_data: List[str] = [], db_type: Literal[-1, 0, 1, 2, 3, 4] = -1, folds: int = 5,
          feature_importance_threshold: float = 0.01, dimensions: Literal[2, 3] = 3, soft_labels: float = 0.0,
          min_samples_required: int = 50, max_num_workers: int = 20, seed: int = 42, parallel: bool = True):
    """Train hierarchical HPO classifiers for the configured dataset setup.

    The function prepares the reference face mesh, loads the modified HPO tree,
    initializes the recursive model structure, builds optional point masks, and
    starts recursive model training across all HPO nodes.

    Args:
        data_dir: Directory containing prepared data files and shared assets.
        out_dir: Output directory for model artifacts, logs, and distributions.
        gmdb_root_dir: Root directory of the GMDB dataset.
        utk_root_dir: Optional root directory of the UTKFace dataset used for negative samples.
        face_region_selections_file: Optional JSON file with manually annotated face-region masks.
        use_face_outline: Whether to include the outer face contour in region masks.
        use_meta_data: List of metadata columns to include as model inputs.
        db_type: Database subtype used to filter or specialize GMDB-HPO training data.
        folds: Number of cross-validation folds.
        feature_importance_threshold: Threshold used to retain important facial points.
        dimensions: Number of face-mesh coordinate dimensions to use.
        soft_labels: Soft-label value applied during dataset balancing.
        min_samples_required: Minimum number of samples required for model training.
        max_num_workers: Maximum number of workers for parallel model processing.
        seed: Random seed used for reproducibility.
        parallel: Whether to enable parallel model creation and processing.
    """
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
    hpo = build_modified_hpo_tree(data_dir, download=False)
    logger.debug(f'Prepared HPO tree and reduced it to the root {hpo.id} with {len(hpo)} nodes.')

    root_model = HumanPhenotypeModel.create_from_hpo(hpo, out_dir, dimensions, use_meta_data, parallel,
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
    """Train one HPO model node and recursively continue with its descendants.

    For each node, the function validates point-mask and sample-count requirements,
    prepares balanced positive and negative datasets, stores distribution summaries,
    trains the current model, and then repeats the process for all child nodes.

    Args:
        pbar: Shared progress bar tracking recursive training status.
        model: Current HPO model node to train.
        db_present: Dataset providing samples where HPO features may be present.
        db_absent: Dataset providing samples used as negative examples.
        point_mask: Current facial point mask used to restrict model input features.
        folds: Number of cross-validation folds.
        seed: Random seed used for reproducibility.
        feature_importance_threshold: Threshold used to retain important facial points.
        soft_labels: Soft-label value applied during dataset balancing.
        available_point_masks: Mapping of HPO term IDs to predefined facial point masks.
        min_samples_required: Minimum number of samples required for model training.
    """
    pbar.set_description_str(model.id)
    os.makedirs(model.log_path, exist_ok=True)

    # if not os.path.exists(model.point_mask_file_output):
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
    logger.debug(
        f'Using {len(db_present_samples.data_df['patient_id'].unique().tolist())} unique patients within {len(db_present_samples.data_df['image_id'].unique().tolist())} images.')
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
    # else:
    #     point_mask = np.load(model.point_mask_file_output)
    update_pbar('Trained', 1, pbar)
    for child_model in model.successors:
        # Continue with next HPO Models
        recursive_model_training(pbar, child_model, db_present, db_absent, point_mask, folds, seed,
                                 feature_importance_threshold, soft_labels, available_point_masks, min_samples_required)


def update_pbar(tag: Literal['Trained', 'NoPoints', 'NoSamples'], value: int, pbar: tqdm.tqdm):
    """Update the progress bar counters and advance the training progress.

    Args:
        tag: Counter name to increment in the progress-bar postfix.
        value: Amount by which the selected counter and progress bar are increased.
        pbar: Active tqdm progress bar instance.
    """
    postfix = json.loads(pbar.postfix)
    postfix[tag] = postfix[tag] + value
    pbar.set_postfix_str(json.dumps(postfix))
    pbar.update(value)


def export_onnx(data_dir: str, out_dir: str, model_dir: str, dimensions: int = 3,
                use_meta_data: list[str] = ['age', 'gender', 'ethnicity'],
                use_face_outline: bool = True, feature_importance_threshold: float = 0.01, soft_labels: float = 0.05,
                seed: int = 42, db_type: int = 4,
                keep_hpos: list[str] = [410030, 160, 2714, 233, 219, 232, 10803, 347, 278, 303, 307, 430, 9928, 9931,
                                        463, 455, 437, 12810, 446, 5280, 3196, 2000, 322, 11829, 275, 12368, 325, 4428,
                                        1999, 280, 4493, 11800, 10669, 293, 294, 290, 341, 348, 2007, 45075, 2223, 2553,
                                        664, 336, 11231, 12745, 494, 508, 581, 286, 100539, 486, 525, 568, 601, 520,
                                        1010, 369, 154, 10805, 12471, 179, 431, 3189, 343, 289, 9890, 337, 574, 637,
                                        582, 316]):
    """Export trained HPO model results to an ONNX-ready JSON representation.

    The function optionally prunes the HPO tree to a selected subset of HPO terms,
    rebuilds the hierarchical model structure for the requested configuration, and
    exports the resulting model metadata and reference mesh to ``model_dir``.

    Args:
        data_dir: Directory containing prepared data files and shared assets.
        out_dir: Root directory containing trained model outputs.
        model_dir: Destination directory for exported ONNX-related model files.
        dimensions: Number of face-mesh coordinate dimensions to use.
        use_meta_data: Metadata fields expected by the exported model.
        use_face_outline: Whether the exported configuration includes face-outline points.
        feature_importance_threshold: Threshold used to retain important facial points.
        soft_labels: Soft-label value used during training for the exported configuration.
        seed: Random seed associated with the trained model version.
        db_type: Database subtype used for the exported training configuration.
        keep_hpos: Optional list of HPO identifiers to retain in the exported hierarchy.
    """
    hpo = build_modified_hpo_tree(data_dir, download=False)

    if keep_hpos:
        keep_hpos = [f'HP:{h:07d}' for h in keep_hpos]
        print(f'Annotated HPO-terms: {len(keep_hpos)}')

        # Step 2: Find all ancestors of leaves to keep
        keep_nodes = set(keep_hpos)
        for leaf_id in keep_hpos:
            node = hpo.find_successor(leaf_id)
            while node and node.predecessor:
                keep_nodes.add(node.predecessor.id)
                node = node.predecessor
        print(f'Keep HPO-terms: {len(keep_nodes)}')

        def prune_subtree(node):
            # Recurse on children first
            node_to_remove = []
            for child in node.successors:
                prune_subtree(child)
                if child.is_leaf() and child.id not in keep_nodes:  # Non-relevant leaf
                    node_to_remove.append(child)

            # Remove unnecessary children
            for child in node_to_remove:
                node.remove_successor(child)

        prune_subtree(hpo)

    reference_face_file = os.path.join(data_dir, 'reference_face.jpg')
    reference_face_mesh = extract_face_meshes([reference_face_file])
    reference_face_mesh = reference_face_mesh[1].reshape((-1, 3))[:, :dimensions]

    version = f'db={db_type}_d={dimensions}_f={use_face_outline}_m=[{"+".join(use_meta_data)}]_t={feature_importance_threshold:.2f}_l={soft_labels:.2f}_s={seed}'

    hpo_model = HumanPhenotypeModel.create_from_hpo(hpo.find_root(), out_dir, dimensions, use_meta_data, False, version,
                                                    8, True)
    HumanPhenotypeModel.export_results_json(hpo_model.find_root(), reference_face_mesh, output_path=model_dir)


if __name__ == '__main__':
    fire.Fire({
        'ablation': ablation_study,
        'train': train,
        'export_onnx': export_onnx,
    })
