"""
This module defines the HumanPhenotypeModel class, which extends HumanPhenotypeTerm
to include deep learning training and prediction capabilities for HPO terms.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.datasets.base_dataset import BaseFaceMeshDataset
from src.hpo_tree.hpo_term import HumanPhenotypeTerm
from src.pl_module import FaceMeshLightningModule
from src.pointnet import ClassificationPointNet
from src.utils.compression import optimize_and_shrink_onnx, deduplicate_onnx_ir_style, compress_onnx_to_gzip
from src.utils.feature_space_investigator import investigate_feature_importance
from src.utils.masking import compute_mask


def create_model(point_mask: ndarray, point_dimensions: int, meta_data: List[str]) -> ClassificationPointNet:
    """
    Creates a PointNet model instance with capacity scaled by the number of active points.

    Args:
        point_mask (ndarray): Boolean mask of active input points.
        point_dimensions (int): Dimensions of the input points.
        meta_data (dict): List of meta data keys.

    Returns:
        ClassificationPointNet: The initialized model.
    """
    if point_mask.sum() > 100:
        base_potence = 6
    elif point_mask.sum() > 50:
        base_potence = 5
    elif point_mask.sum() > 25:
        base_potence = 4
    else:
        base_potence = 3
    return ClassificationPointNet(num_classes=1, dropout=0.5, point_dimension=point_dimensions,
                                  base_potence=base_potence, meta_data=meta_data)


class BatchConfig(Enum):
    """
    Enum for defining batch size configurations based on dataset size.
    """
    BATCH_SIZE_XS = 1
    BATCH_SIZE_S = 4
    BATCH_SIZE_M = 16
    BATCH_SIZE_L = 64
    BATCH_SIZE_XL = 128

    @staticmethod
    def get_batch_size(dataset_size: int) -> int:
        """
        Determines the appropriate batch size based on the number of samples in the dataset.

        Args:
            dataset_size (int): The number of samples in the dataset.

        Returns:
            int: The selected batch size.
        """
        if dataset_size < 50:
            return BatchConfig.BATCH_SIZE_XS.value
        elif dataset_size < 200:
            return BatchConfig.BATCH_SIZE_S.value
        elif dataset_size < 1000:
            return BatchConfig.BATCH_SIZE_M.value
        elif dataset_size < 5000:
            return BatchConfig.BATCH_SIZE_L.value
        return BatchConfig.BATCH_SIZE_XL.value


class HumanPhenotypeModel(HumanPhenotypeTerm):
    """
    A class representing an HPO term associated with a deep learning model.

    This class extends HumanPhenotypeTerm by adding methods for training,
    predicting, and exporting models (PointNet) related to specific HPO phenotypes.
    """

    def __init__(self, hpo: HumanPhenotypeTerm, output_dir: str, dimensions: int, meta_data: List[str], parallel: bool,
                 max_num_workers: int, version: str):
        """
        Initializes a HumanPhenotypeModel.

        Args:
            hpo (HumanPhenotypeTerm): The HPO term object to base this model on.
            output_dir (str): The root directory for saving model outputs and logs.
            dimensions (int): The dimensionality of the input points (e.g., 2 or 3).
            meta_data (List[str]): List of metadata fields to include in the model.
            parallel (bool): Whether to use parallel processing for training folds.
            max_num_workers (int): Maximum number of workers for data loading and parallel execution.
            version (str): Version identifier for the model.
        """
        super().__init__(hpo._id, hpo._name, hpo._definition, hpo._comment)
        self._successors = []
        self._parent = None
        self.output_dir = output_dir
        self.dimensions = dimensions
        self.meta_data = meta_data
        self.parallel = parallel
        self.max_num_workers = max_num_workers
        self.version = version
        self.log_name = self.id.replace(':', '_')
        self.log_path = os.path.join(self.output_dir, self.log_name, self.version)
        self.global_fi_csv_file = os.path.join(self.log_path, 'global_feature_importance.csv')
        self.global_fi_file = os.path.join(self.log_path, 'global_feature_importance.npy')
        self.val_metrics_file = os.path.join(self.log_path, 'validation_metrics_kfolds.csv')
        self.dist_syndrome_file = os.path.join(self.log_path, 'distributions_syndrome.csv')
        self.dist_hpo_present_file = os.path.join(self.log_path, 'distributions_hpo_present.csv')
        self.dist_hpo_absent_file = os.path.join(self.log_path, 'distributions_hpo_absent.csv')
        self.point_mask_file_input = os.path.join(self.log_path, 'point_mask_input.npy')
        self.point_mask_file_output = os.path.join(self.log_path, 'point_mask_output.npy')

    def is_trained(self):
        """
        Checks if the model for this HPO term has been trained.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        return os.path.exists(self.global_fi_file)

    def is_parent_node(self):
        """
        Checks if this node acts as a parent to any trained successor nodes.

        Returns:
            bool: True if at least one successor is trained.
        """
        return len([s for s in self.successors if s.is_trained()]) > 0

    def is_compression_node(self):
        """
        Checks if this node has exactly one trained successor node.

        Returns:
            bool: True if exactly one successor is trained.
        """
        return len([s for s in self.successors if s.is_trained()]) == 1

    def list_leaf_nodes(self):
        """
        Lists all leaf nodes in the subtree that have been trained.

        Returns:
            List[HumanPhenotypeModel]: List of trained leaf nodes.
        """
        return [s for s in self.all_successors() if s.is_leaf() and s.is_trained()]

    def list_parent_nodes(self):
        """
        Lists all parent nodes in the subtree that have been trained.

        Returns:
            List[HumanPhenotypeModel]: List of trained parent nodes.
        """
        return [p for p in self.all_successors() if p.is_parent_node() and p.is_trained()]

    def list_compression_nodes(self):
        """
        Lists all compression nodes in the subtree that have been trained.

        Returns:
            List[HumanPhenotypeModel]: List of trained compression nodes.
        """
        return [p for p in self.all_successors() if p.is_compression_node() and p.is_trained()]

    def train(self, dataset: BaseFaceMeshDataset, folds: int, feature_importance_threshold: float, seed: int,
              point_mask: np.ndarray):
        """
        Trains the model for this HPO term using k-fold cross-validation.

        Args:
            dataset (BaseFaceMeshDataset): The dataset to train on.
            folds (int): Number of folds for cross-validation.
            feature_importance_threshold (float): Threshold for selecting points for successor models.
            seed (int): Random seed for reproducibility.
            point_mask (np.ndarray): Boolean mask indicating which points to use as input.

        Returns:
            np.ndarray: The resulting point mask for successor models.
        """
        os.makedirs(self.log_path, exist_ok=True)

        with open(os.path.join(self.log_path, 'config.json'), 'w') as f:
            json.dump({
                'dimensions': self.dimensions,
                'meta_data': self.meta_data,
                'parallel': self.parallel,
                'feature_importance_threshold': feature_importance_threshold,
                'seed': seed,
            }, f)

        logger.debug(f'Training model for {self.id} on dataset with {len(dataset)} samples in total...')
        kfold_feature_importances = []
        kfold_validation_metrics = []

        np.save(self.point_mask_file_input, point_mask)

        if not self.is_trained():
            logger.debug(f'Creating a {folds}-fold CV split...')
            kfold_datasets = dataset.kfold_data(folds, dataset.label_column, 'patient_id', random_state=seed)

            if self.parallel and len(kfold_datasets) > 1:
                # Use ProcessPoolExecutor for parallel fold training
                with ProcessPoolExecutor(mp_context=torch.multiprocessing.get_context('spawn')) as executor:
                    futures = []
                    logger.debug(f'Running the {folds} fold CV in parallel...')
                    for fold, (train_dataset, val_dataset) in enumerate(kfold_datasets):
                        futures.append(
                            executor.submit(
                                self._train_fold,
                                fold, train_dataset, val_dataset, point_mask, self.max_num_workers // folds
                            )
                        )

                    for future in as_completed(futures):
                        log_fold_path, feature_importance, best_epoch = future.result()
                        kfold_feature_importances.append(feature_importance.sum(axis=0))

                        df = pd.read_csv(os.path.join(log_fold_path, 'validation_metrics.csv'))
                        to_dict = df[df["epoch"] == best_epoch].drop(columns=['epoch']).iloc[0].to_dict()
                        kfold_validation_metrics.append(to_dict)
            else:
                logger.debug(f'Running the {folds} fold CV in serial...')
                for fold, (train_dataset, val_dataset) in enumerate(kfold_datasets):
                    log_fold_path, feature_importance, best_epoch = self._train_fold(fold, train_dataset,
                                                                                     val_dataset, point_mask,
                                                                                     self.max_num_workers // folds)
                    kfold_feature_importances.append(feature_importance.sum(axis=0))

                    df = pd.read_csv(os.path.join(log_fold_path, 'validation_metrics.csv'))
                    to_dict = df[df["epoch"] == best_epoch].drop(columns=['epoch']).iloc[0].to_dict()
                    kfold_validation_metrics.append(to_dict)

            # Compute the validation metrics with std over all folds
            kfold_validation_metrics = pd.DataFrame(kfold_validation_metrics)
            kfold_validation_metrics['support'] = len(dataset)
            kfold_validation_metrics.to_csv(self.val_metrics_file, index=False)

            # Visualize feature importance and metrics
            kfold_feature_importances = np.asarray(kfold_feature_importances)
            np.save(self.global_fi_file, kfold_feature_importances)

        logger.debug('Finished training all folds.')
        kfold_feature_importances = np.load(self.global_fi_file)
        # Filter points based on their feature importance
        logger.debug('Creating point mask for successors...')
        importances_sum = kfold_feature_importances.sum(axis=0)
        importances_sum = MinMaxScaler().fit_transform(importances_sum.reshape(-1, 1)).flatten()
        feature_importances_indices = np.where(point_mask == True)
        global_fi_df = pd.DataFrame(importances_sum, index=feature_importances_indices[0]).transpose()
        global_fi_df.to_csv(self.global_fi_csv_file, index=False)
        point_mask = importances_sum >= feature_importance_threshold
        point_mask = compute_mask(dataset.mask_pts, point_mask)
        np.save(self.point_mask_file_output, point_mask)
        logger.debug('Finished creating point mask for successors.')

        HumanPhenotypeModel.export_results_json(self, dataset.reference_mesh)
        logger.debug(f'Results exported to {os.path.join(self.find_root().log_path, "result.json")}')

        return point_mask

    def _train_fold(self, fold, train_dataset: BaseFaceMeshDataset, val_dataset: BaseFaceMeshDataset,
                    point_mask: np.ndarray, num_workers: int):
        """
        Internal method to train a single cross-validation fold.

        Args:
            fold (int): The fold index.
            train_dataset (BaseFaceMeshDataset): Training data for this fold.
            val_dataset (BaseFaceMeshDataset): Validation data for this fold.
            point_mask (np.ndarray): Boolean mask for input points.
            num_workers (int): Number of workers for data loading.

        Returns:
            tuple: (output_dir, feature_importance, best_epoch)
        """
        fold_folder = f'fold_{fold}'
        output_dir = os.path.join(self.log_path, fold_folder)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_files = self.find_ckpt_files(output_dir)
        val_met_file = os.path.join(output_dir, 'validation_metrics.csv')
        global_feature_importance_file = os.path.join(output_dir, 'global_feature_importance.npy')
        train_dataset.data_df.to_csv(os.path.join(output_dir, 'train_set.csv'))
        val_dataset.data_df.to_csv(os.path.join(output_dir, 'val_set.csv'))

        # sample_face_mesh = train_dataset.reference_mesh if point_mask is None else train_dataset.reference_mesh[
        #     point_mask]

        checkpoint_file = None
        if len(checkpoint_files) > 0:
            logger.debug(f'Found checkpoints: {checkpoint_files}')
            checkpoint_file = checkpoint_files[-1]
            logger.debug(f'Selected checkpoint: {checkpoint_file}')

        batch_size = BatchConfig.get_batch_size(len(train_dataset))
        # Decide for a good number of workers to load the data according to the batch size
        max_workers = num_workers if self.parallel else 0
        persistent_workers = max_workers > 0
        num_workers_train = min(max(len(train_dataset) // batch_size, 1), max_workers)
        batch_size_train = min(len(train_dataset), batch_size)
        mp_context = torch.multiprocessing.get_context('spawn')
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                  shuffle=True, num_workers=num_workers_train, persistent_workers=persistent_workers,
                                  multiprocessing_context=mp_context)
        num_workers_val = min(max(len(val_dataset) // batch_size, 1), max_workers)
        batch_size_val = min(len(train_dataset), 1024)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                                num_workers=num_workers_val, persistent_workers=persistent_workers,
                                multiprocessing_context=mp_context)
        logger.debug(
            f'{fold}-Fold with num_workers (train: {num_workers_train}, val: {num_workers_val}) and batch size (train: {batch_size_train}, val: {batch_size_val}).')

        model = create_model(point_mask, self.dimensions, self.meta_data)

        if checkpoint_file:
            lightning_module = FaceMeshLightningModule.load_from_checkpoint(checkpoint_file, model=model,
                                                                            map_location='cpu')
        else:
            optimizer_config = {'lr': 1e-4}
            scheduler_config = {'mode': 'min', 'patience': 5}

            lightning_module = FaceMeshLightningModule(
                model=model,
                num_classes=1,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                monitor='val_loss',
            )

        if not os.path.exists(global_feature_importance_file):
            tb_logger = TensorBoardLogger(self.output_dir, name=self.log_name,
                                          sub_dir=fold_folder, version=self.version)
            checkpoint = ModelCheckpoint(dirpath=tb_logger.log_dir, save_top_k=1, monitor='val_loss', mode='min',
                                         every_n_epochs=1)
            callbacks = [
                # ModelSummary(max_depth=1),
                EarlyStopping('val_loss', mode='min', patience=5),
                checkpoint
            ]
            tmp = os.path.join(self.output_dir, 'lightning_tmp')
            os.makedirs(tmp, exist_ok=True)
            trainer = Trainer(max_epochs=25, accelerator='gpu', logger=tb_logger, callbacks=callbacks,
                              log_every_n_steps=1, default_root_dir=tmp)
            trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=checkpoint_file)

            df = pd.DataFrame(lightning_module.metrics_history)
            if os.path.exists(val_met_file):
                df_previous = pd.read_csv(val_met_file)
                df = pd.concat([df_previous, df], ignore_index=True)
            df.to_csv(val_met_file, index=False)

        feature_importance = investigate_feature_importance(output_dir, lightning_module,
                                                            val_dataset, self.parallel)

        checkpoint_file = self.find_ckpt_files(output_dir)[-1]
        return output_dir, feature_importance, torch.load(checkpoint_file)["epoch"]

    def predict(self, facemeshes: torch.Tensor, use_metric: Literal[
        'all', 'accuracy', 'auroc', 'f1_score', 'jaccard_index', 'matthews_corrcoef', 'precision', 'recall'] = 'all',
                **kwargs):
        """
        Performs inference using the trained models.

        Args:
            facemeshes (torch.Tensor): Input face mesh tensors (Batch, Dimensions, Points).
            use_metric (Literal['all', 'accuracy', 'auroc', 'f1_score', 'jaccard_index', 'matthews_corrcoef', 'precision', 'recall'], optional): Whether to use the best fold or an ensemble of all folds. Defaults to 'best'.
            **kwargs: Additional arguments passed to the model's predict method.

        Returns:
            dict: A dictionary mapping HumanPhenotypeModel nodes to their prediction results.
        """
        # assert self.is_trained(), f'{self} has not been trained yet.'
        if not self.is_trained():
            logger.warning(f'{str(self)} is not trained yet.')
            return {self: np.asarray([np.asarray([-1], dtype=np.float32) for _ in range(len(facemeshes))])}

        fold_dir = [fold_dir for fold_dir in os.listdir(self.log_path) if fold_dir.startswith('fold_')]
        fold_model_ckpts = [self.find_ckpt_files(os.path.join(self.log_path, fold))[-1] for fold in fold_dir]
        point_mask = np.load(self.point_mask_file_input)
        batch_size = facemeshes.shape[0]
        reduced_facemeshes = facemeshes.reshape(batch_size, -1, self.dimensions)
        reduced_facemeshes = reduced_facemeshes[:,point_mask]
        reduced_facemeshes = reduced_facemeshes.reshape(batch_size, self.dimensions, -1)
        result = {}
        if use_metric == 'all':
            pl_models = [FaceMeshLightningModule.load_from_checkpoint(ckpt, model=create_model(point_mask,
                                                                                                    self.dimensions,
                                                                                                    self.meta_data),
                                                                      map_location='cpu') for ckpt in
                         fold_model_ckpts]
            all_result_preds = []
            for idx, pl_model in enumerate(pl_models):
                pl_model.eval()
                reduced_facemeshes.to(pl_model.device)
                for key, value in kwargs.items():
                    value.to(pl_model.device)
                pred = pl_model.predict(reduced_facemeshes, **kwargs)
                all_result_preds.append(pred.detach().cpu().numpy())
            result.update({self: np.asarray(all_result_preds).squeeze()})
            for successor in self.successors:
                s_result = successor.predict(facemeshes, use_metric=use_metric, **kwargs)
                result.update(s_result)
        else:
            all_model_performances = pd.read_csv(self.val_metrics_file)
            best_model_idx = all_model_performances[use_metric].argmax()
            pl_model = FaceMeshLightningModule.load_from_checkpoint(fold_model_ckpts[best_model_idx],
                                                                    model=create_model(point_mask, self.dimensions,
                                                                                            self.meta_data),
                                                                    map_location='cpu')
            pl_model.eval()
            reduced_facemeshes.to(pl_model.device)
            for key, value in kwargs.items():
                value.to(pl_model.device)
            pred = pl_model.predict(reduced_facemeshes, **kwargs)
            result.update({self: pred.detach().cpu().numpy()})
            for successor in self.successors:
                s_result = successor.predict(facemeshes, use_metric=use_metric, **kwargs)
                result.update(s_result)
        return result

    def export_to_onnx(self, export_dir: str, recursive: bool = False, use_metric: Literal[
        'accuracy', 'auroc', 'f1_score', 'jaccard_index', 'matthews_corrcoef', 'precision', 'recall'] = 'matthews_corrcoef', optimize: bool = True):
        """
        Exports the best fold model to ONNX format.

        Args:
            export_dir (str): Directory where the ONNX file should be saved.
            recursive (bool): Set to True to export all successor models.
        """
        if os.path.exists(self.log_path):
            fold_dir = [fold_dir for fold_dir in os.listdir(self.log_path) if fold_dir.startswith('fold_')]
            fold_model_ckpts = [self.find_ckpt_files(os.path.join(self.log_path, fold))[-1] for fold in fold_dir]
            point_mask = np.load(self.point_mask_file_input)
            all_model_performances = pd.read_csv(self.val_metrics_file)
            best_model_idx = all_model_performances[use_metric].argmax()
            pl_model = FaceMeshLightningModule.load_from_checkpoint(fold_model_ckpts[best_model_idx],
                                                                    model=create_model(point_mask, self.dimensions,
                                                                                            self.meta_data),
                                                                    map_location='cpu')
            data_structure = {}
            for meta_item in self.meta_data:
                data_structure.update({meta_item: torch.randn(2)})
            input_data = (torch.randn(2, self.dimensions, 478)[:, :, point_mask], data_structure)
            input_names = ["input"] + list(data_structure.keys())
            pl_model.eval()
            onnx_file = os.path.join(export_dir, f"{self.id.replace(':', '_')}.onnx")
            pl_model.to_onnx(
                onnx_file,
                input_data,
                export_params=True,
                opset_version=22,
                input_names=input_names,
                output_names=["logits", "features"],
                external_data=False,
                # dynamic_axes={name: {0: "batch"} for name in input_names}
            )
            if optimize:
                onnx_optimized_file = onnx_file.replace(".onnx", "_optimized.onnx")
                optimize_and_shrink_onnx(onnx_file, onnx_optimized_file)
                onnx_ir_file = onnx_file.replace(".onnx", "_ir.onnx")
                deduplicate_onnx_ir_style(onnx_optimized_file, onnx_ir_file)
                onnx_gzip_file = f'{onnx_file}.gzip'
                compress_onnx_to_gzip(onnx_ir_file, onnx_gzip_file)
                # os.remove(onnx_file)  # Keep *.onnx file
                os.remove(onnx_optimized_file)
                os.remove(onnx_ir_file)
            if recursive:
                for successor in self.successors:
                    successor.export_to_onnx(export_dir, recursive=recursive, optimize=optimize)

    def export_result_dict(self):
        """
        Creates a dictionary containing model configuration, metrics, and importance values.

        Returns:
            dict: The result dictionary.
        """
        result = {
            "id": self.id,
            "parent": self.predecessor.id if self.predecessor else None,
            "description": self.name,
            "definition": self.definition,
            "comment": self.comment,
            "metrics": None,
            "importance_values": None,
            "database": {
                "hpo_present": None,
                "hpo_absent": None,
                "syndromes": None,
            }
        }
        config_file = os.path.join(self.log_path, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                result.update(config)
        if os.path.exists(self.global_fi_csv_file):
            result['metrics'] = pd.read_csv(self.val_metrics_file).to_dict()
            result['importance_values'] = pd.read_csv(self.global_fi_csv_file).to_dict()
            result['database']['hpo_present'] = pd.read_csv(self.dist_hpo_present_file).to_dict()
            result['database']['hpo_absent'] = pd.read_csv(self.dist_hpo_absent_file).to_dict()
            result['database']['syndromes'] = pd.read_csv(self.dist_syndrome_file).to_dict()
        return result

    @staticmethod
    def find_ckpt_files(output_dir: str):
        """
        Finds all checkpoint (.ckpt) files in the specified directory.

        Args:
            output_dir (str): Directory to search.

        Returns:
            List[str]: List of paths to checkpoint files.
        """
        if not os.path.exists(output_dir):
            return []
        files = list(filter(lambda f: f.endswith('.ckpt'), os.listdir(output_dir)))
        return [os.path.join(output_dir, ckpt_name) for ckpt_name in files]

    @staticmethod
    def create_from_hpo(hpo: HumanPhenotypeTerm, output_dir: str, dimensions: int, meta_data: List[str], parallel: bool,
                        version: str, max_num_workers: int, recursive: bool = False):
        """
        Factory method to create a HumanPhenotypeModel (and optionally its children) from a HumanPhenotypeTerm.

        Args:
            hpo (HumanPhenotypeTerm): The base HPO term.
            output_dir (str): Output directory for the model.
            dimensions (int): Point dimensionality.
            meta_data (List[str]): Metadata fields.
            parallel (bool): Whether to use parallel processing.
            version (str): Model version.
            max_num_workers (int): Max workers.
            recursive (bool, optional): Whether to create models for all successors recursively. Defaults to False.

        Returns:
            HumanPhenotypeModel: The created model node.
        """
        hpo_model = HumanPhenotypeModel(hpo, output_dir, dimensions, meta_data, parallel, max_num_workers, version)
        if recursive:
            for child_hpo in hpo.successors:
                model = HumanPhenotypeModel.create_from_hpo(child_hpo, output_dir, dimensions, meta_data, parallel,
                                                            version, max_num_workers, recursive)
                hpo_model.add_successor(model)
        return hpo_model

    @staticmethod
    def export_results_json(model: "HumanPhenotypeModel", reference_mesh: np.ndarray, output_dir: str = None):
        """
        Exports the entire HPO model tree structure and results to a JSON file.

        Args:
            model (HumanPhenotypeModel): The model node to start export from.
            reference_mesh (np.ndarray): The reference face mesh used for visualization.
            output_dir (str): Output directory for the result json file.
        """
        root = model.find_root()
        os.makedirs(root.log_path, exist_ok=True)
        export_path = os.path.join(output_dir if output_dir else root.log_path, 'result.json')
        data = {
            "reference_mesh": [{
                                   "id": idx,
                                   "x": reference_mesh[idx][0],
                                   "y": reference_mesh[idx][1],
                                   "z": reference_mesh[idx][2]
                               } if root.dimensions == 3 else {
                "id": idx,
                "x": reference_mesh[idx][0],
                "y": reference_mesh[idx][1]
            } for idx in range(reference_mesh.shape[0])],
            "nodes": [],
            "edges": [],
            "root": root.id
        }

        nodes = []
        edges = []

        def recursive_node_extractor(node: "HumanPhenotypeModel"):
            if node.is_trained():
                nodes.append(node.export_result_dict())
                if node.predecessor:
                    edge = {'source': node.predecessor.id, 'target': node.id}
                    if len(list(filter(lambda n: n['source'] == node.id and n['target'] == child.id, edges))) == 0:
                        edges.append(edge)
                for child in node.successors:
                    recursive_node_extractor(child)

        recursive_node_extractor(root)

        data['nodes'] = nodes
        data['edges'] = edges

        json.dump(data, open(export_path, "w"), indent=2)
        logger.debug(f"HPO tree data exported to {export_path}")
