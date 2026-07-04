import inspect
import itertools
import os
from abc import ABC, abstractmethod
from typing import Literal, List

import numpy as np
import pandas as pd
import torch
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, KMeansSMOTE, BorderlineSMOTE, SMOTEN, SMOTENC, \
    SVMSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour
from loguru import logger
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, IncrementalPCA, KernelPCA, LatentDirichletAllocation, \
    MiniBatchDictionaryLearning, MiniBatchNMF, MiniBatchSparsePCA, NMF, SparseCoder, SparsePCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import Dataset
from typing_extensions import Self

ethnicity_mapping = {
    'European': 0,
    'Asian': 1,
    'African': 2,
    'Others': 3,
    'Unknown': -1
}

gender_mapping = {
    'female': 0,
    'male': 1,
    'diverse': 2,
    'unknown': -1
}


class BaseFaceMeshDataset(Dataset, ABC):
    """
    Base class for face mesh datasets.
    Provides common functionality for loading, preprocessing, sampling, and transforming face mesh data.
    """

    def __init__(self, label_column: str, root_dir: str, data_file: str, data_df: pd.DataFrame = None,
                 reference_mesh: np.ndarray = None, dimensions: int = 3, transform=None):
        """
        Initialize the BaseFaceMeshDataset.
        :param label_column: Name of the column containing the labels.
        :param root_dir: Root directory for the dataset.
        :param data_file: Path to the CSV file containing the data.
        :param data_df: Optional pre-loaded DataFrame. If provided, data_file is ignored for loading.
        :param reference_mesh: Optional reference mesh for Procrustes alignment.
        :param dimensions: Number of dimensions for the points (e.g., 2 or 3).
        :param transform: Optional transformation to apply to the points.
        """
        super().__init__()
        self.label_column = label_column
        self.root_dir = root_dir
        self._reference_mesh = reference_mesh
        self.mask_pts = None
        self.dimensions = dimensions
        self.transform = transform
        if data_df is not None:
            self.data_df = data_df
        elif isinstance(data_file, str) and os.path.exists(data_file):
            self.data_df = pd.read_csv(data_file, low_memory=False)
        else:
            logger.debug('Preprocessing dataset...')
            self.data_df = self._load_data(data_file, reference_mesh)
            logger.debug(f'Dataset is ready under {data_file} and will also be used in future runs.')
        self.data_df.dropna(subset=self.data_columns(), inplace=True)

    @property
    def reference_mesh(self) -> np.ndarray:
        """
        Get the reference mesh used for alignment.
        If not explicitly set, uses the first entry in the dataset.
        :return: Reference mesh as a numpy array.
        """
        if self._reference_mesh is None:
            return self.data_df[self.data_columns()].to_numpy()[0]
        return self._reference_mesh

    @abstractmethod
    def data_columns(self) -> List[str]:
        """
        Abstract method to return the list of columns containing point data.
        :return: List of column names.
        """
        pass

    @abstractmethod
    def label_columns(self) -> List[str]:
        """
        Abstract method to return the list of possible label columns.
        :return: List of column names.
        """
        pass

    @abstractmethod
    def _load_data(self, data_file: str, reference_mesh: np.ndarray = None) -> pd.DataFrame:
        """
        Abstract method to load and preprocess data if not already available in a CSV.
        :param data_file: Path where the processed data should be saved/loaded from.
        :param reference_mesh: Optional reference mesh for alignment during loading.
        :return: Processed DataFrame.
        """
        pass

    @abstractmethod
    def _create_new_dataset(self, new_data_df: pd.DataFrame) -> Self:
        """
        Abstract method to create a new instance of the dataset with a filtered/modified DataFrame.
        :param new_data_df: The new DataFrame to use.
        :return: A new instance of the dataset class.
        """
        pass

    def set_point_mask(self, mask: np.ndarray) -> None:
        """
        Define a mask that removes points from the mesh.
        :param mask: Boolean or index mask to apply to the points.
        """
        self.mask_pts = mask

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Loads points, performs alignment, applies transformations and masks.
        :param idx: Index of the item.
        :return: Dictionary containing 'x' (points), 'y' (label), 'age', 'gender', 'ethnicity'.
        """
        # Standardized: always loads 478 x 3 numpy array and optional features
        pts = self.data_df[self.data_columns()].iloc[idx].to_numpy().reshape(-1, 3)
        pts = pts[:, :self.dimensions]
        if self.transform:
            pts = self.transform(pts)
        if self.mask_pts is not None:
            pts = pts[self.mask_pts]
        pts = pts.reshape(self.dimensions, -1)
        pts = pts[:self.dimensions, :]
        label = self.data_df[self.label_column].iloc[idx]
        age = self.data_df['age'].fillna(value=-1).iloc[idx]
        gender = gender_mapping[self.data_df['gender'].fillna(value=-1).iloc[idx]]
        ethnicity = ethnicity_mapping[self.data_df['ethnicity'].fillna(value=-1).iloc[idx]]
        return {
            'x': torch.tensor(pts, dtype=torch.float),
            'y': torch.tensor(label, dtype=torch.float),
            'age': torch.tensor(age, dtype=torch.float),
            'gender': torch.tensor(gender, dtype=torch.float),
            'ethnicity': torch.tensor(ethnicity, dtype=torch.float),
        }

    def __len__(self):
        """
        Return the number of items in the dataset.
        :return: Length of the dataset.
        """
        return len(self.data_df)

    def concat(self, dataset, label_value: float = 0) -> Self:
        """
        Concatenate another dataset to this one.
        Handles missing label columns and patient IDs.
        :param dataset: The dataset to concatenate.
        :param label_value: Default label value for the added dataset if it lacks the label column.
        :return: A new dataset instance containing combined data.
        """
        if self.label_column not in dataset.data_df.columns:
            dataset.data_df[self.label_column] = [label_value] * len(dataset.data_df)
        if 'patient_id' not in dataset.data_df.columns:
            first_non_patient_id = 999999
            dataset.data_df['patient_id'] = list(
                range(first_non_patient_id, first_non_patient_id + len(dataset.data_df)))
        combined_df = pd.concat([self.data_df, dataset.data_df], ignore_index=True)
        count_hpo_present = len(combined_df[combined_df[self.label_column] == 1])
        count_hpo_absent = len(combined_df[combined_df[self.label_column] == label_value])
        logger.debug(f'Dataset: hpo_present={count_hpo_present}, hpo_absent={count_hpo_absent}')
        return self._create_new_dataset(combined_df)

    def sample(self, n: int, random_state: int) -> Self:
        """
        Randomly sample n items from the dataset.
        :param n: Number of items to sample.
        :param random_state: Random seed.
        :return: A new dataset instance with sampled data.
        """
        if n > len(self.data_df):
            samples = pd.concat(
                [self.data_df, self.data_df.sample(n=n - len(self.data_df), random_state=random_state, replace=True)],
                ignore_index=True)
        else:
            samples = self.data_df.sample(n=n, random_state=random_state, replace=False)
        return self._create_new_dataset(samples)

    def sample_by_distribution(self, n: int, random_state: int, distribution) -> Self:
        """
        Sample n items from the dataset following a target distribution for age, ethnicity, and gender.
        Uses a stratified sampling approach to match the joint distribution under independence assumption.
        :param n: Number of items to sample.
        :param random_state: Random seed.
        :param distribution: Target distribution dictionary with 'age', 'ethnicity', and 'gender' keys.
        :return: A new dataset instance with sampled data.
        """
        logger.debug(f'Sampling a dataset distribution with {n} samples...')
        # Calculate total counts per category for normalization
        total_counts = {
            key: sum(count for _, count in vals.items())
            for key, vals in distribution.items()
        }

        # Normalize each category to get distribution
        distributions = {
            key: {str(cat): count / total_counts[key] for cat, count in vals.items()}
            for key, vals in distribution.items()
        }

        # Create all permutations
        permutations = list(itertools.product(
            distributions['age'].keys(),
            distributions['ethnicity'].keys(),
            distributions['gender'].keys()
        ))

        # Compute joint distribution under assumption of independence:
        permutation_distribution = {}
        for age, eth, gen in permutations:
            key = f"{gen}_{age}_{eth}"
            prob = distributions['age'][age] * distributions['ethnicity'][eth] * distributions['gender'][gen]
            permutation_distribution[key] = prob

        # Example: Your original DataFrame is df
        # Your stratification columns list (keys in series_dict)
        stratify_cols = list(total_counts.keys())

        # Check if the stratify columns exist in your df
        assert all(col in self.data_df.columns for col in stratify_cols), "Some stratify columns missing in df"

        # Create a single combined stratification key by concatenating the values
        df = self.data_df.copy()
        df['_stratify_key'] = df[stratify_cols].astype(str).agg('_'.join, axis=1)
        existing_keys = df['_stratify_key'].value_counts().to_dict()

        total_prob = 0.0
        valid_keys = {}

        # Filter keys with count >0 and compute sum of their probs
        for key, prob in permutation_distribution.items():
            if key in existing_keys and existing_keys[key] > 0:
                valid_keys[key] = prob
                total_prob += prob

        if len(valid_keys) == 0:
            # Fallback: Distribute all keys equally
            equal_probability = 1.0 / len(existing_keys)
            valid_keys = {key: equal_probability for key in existing_keys.keys()}
            total_prob = 1.0

        # Minimum-Sampling Strategie
        num_valid_categories = len(valid_keys)
        if num_valid_categories > n:
            valid_df = pd.DataFrame({'key': valid_keys.keys(), 'prob': valid_keys.values()})
            top_n_valid_df = valid_df.nlargest(n, 'prob')
            valid_keys = dict(zip(top_n_valid_df['key'], top_n_valid_df['prob']))
            total_prob = sum(valid_keys.values())
            min_samples_per_cat = 1
            remaining_samples = 0
        else:
            min_samples_per_cat = max(1, n // (num_valid_categories * 2))  # Minimum of one per category
            remaining_samples = n - (min_samples_per_cat * num_valid_categories)

        # Renormalize valid keys probabilities to sum to 1
        for key in valid_keys:
            valid_keys[key] /= total_prob

        sample_collection = []

        grouped = {k: g for k, g in df.groupby('_stratify_key', sort=False) if len(g) > 0}

        # Phase 1: Minimum-Samples per Category
        for key in valid_keys:
            if key in grouped:
                filtered_df = grouped.get(key)
                count = len(filtered_df)
                sample_size = min(min_samples_per_cat, count)
                if sample_size > 0:
                    sample = filtered_df.sample(n=sample_size, replace=False, random_state=random_state)
                    sample_collection.append(sample)

        # Phase 2: Distribute the remaining samples
        if remaining_samples > 0 and sample_collection:
            # Compute additional samples proportional
            additional_probs = {k: v * remaining_samples for k, v in valid_keys.items()}

            for key, add_size in additional_probs.items():
                if key in grouped:
                    filtered_df = grouped.get(key)
                    count = len(filtered_df)
                    sample_size = int(round(add_size))

                    if sample_size > 0 and count > 0:
                        sample_size = min(sample_size, count)  # Don't use more than available
                        sample = filtered_df.sample(n=sample_size, replace=False, random_state=random_state + 42)
                        sample_collection.append(sample)

        # FALLBACKS
        if len(sample_collection) == 0:
            # No valid categories → random sample
            sampled_df = df.sample(n=min(n, len(df)), replace=True, random_state=random_state)
            logger.warning(f'No valid stratified samples → using {len(sampled_df)} random samples (requested: {n})')
        else:
            sampled_df = pd.concat(sample_collection, ignore_index=True)

            # Final Check: If still not enough, fill up
            if len(sampled_df) < n:
                missing = n - len(sampled_df)
                used_images = sampled_df['image_id'].tolist()
                extra_samples = df[~df['image_id'].isin(used_images)].sample(n=missing,
                                                                             replace=True,
                                                                             random_state=random_state)
                sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)
                logger.debug(f'Filled up: {len(sampled_df)} Samples (requested: {n})')

        # Drop helper column
        sampled_df = sampled_df.drop(columns=['_stratify_key'], errors='ignore')
        logger.debug(f'Sampled {len(sampled_df)} rows from {len(valid_keys)} categories (requested: {n})')
        return self._create_new_dataset(sampled_df)

    def sample_from_label_column(self, n: int, label_value: int, random_state: int) -> Self:
        """
        Sample n items where the label matches a specific value.
        :param n: Number of items to sample. If < 0, all matching items are returned (shuffled).
        :param label_value: The label value to filter by.
        :param random_state: Random seed.
        :return: A new dataset instance with sampled data.
        """
        data_with_label_value = self.data_df[self.data_df[self.label_column] == label_value]
        if n < 0:
            samples = data_with_label_value.sample(frac=1, random_state=random_state, replace=False)
        else:
            # samples = data_with_label_value.sample(n=n, random_state=random_state, replace=True)
            if n > len(data_with_label_value):
                samples = pd.concat([data_with_label_value,
                                     data_with_label_value.sample(n=n - len(data_with_label_value),
                                                                  random_state=random_state, replace=True)],
                                    ignore_index=True)
            else:
                samples = data_with_label_value.sample(n=n, random_state=random_state, replace=False)
        return self._create_new_dataset(samples)

    def extract_distributions(self, columns: List[str]):
        """
        Extract the distribution of values for specific columns.
        :param columns: List of column names to analyze.
        :return: Dictionary mapping column names to Pandas Series of value counts.
        """
        counts = {column: self.data_df[column].value_counts(dropna=False) for column in columns}
        series_dict = {}
        for col, vals in counts.items():
            index = [v[0] for v in vals.items()]
            data = [v[1] for v in vals.items()]
            series_dict[col] = pd.Series(data=data, index=index).sort_index()
            series_dict[col].index = series_dict[col].index.where(series_dict[col].index.notna(), other='Missing')
        return series_dict

    def split_data(self, test_size: float, random_state: int) -> tuple[Self, Self]:
        """
        Split the dataset into training and testing sets.
        Uses stratified splitting based on the label column.
        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Random seed.
        :return: Tuple of (train_dataset, test_dataset).
        """
        # Initialize an empty column for split
        if not 'Split' in self.data_df.columns:
            self.data_df['Split'] = -1
            train_idx, test_idx = train_test_split(
                self.data_df[self.data_columns()].index,
                stratify=self.data_df[self.label_column],
                test_size=test_size,
                random_state=random_state
            )
            self.data_df.loc[train_idx, 'Split'] = 0
            self.data_df.loc[test_idx, 'Split'] = 1

        train_data_df = self.data_df[self.data_df['Split'] == 0]
        val_data_df = self.data_df[self.data_df['Split'] == 1]

        return self._create_new_dataset(train_data_df), self._create_new_dataset(val_data_df)

    def kfold_data(self, folds: int, label, group_label, random_state: int) -> List[tuple[Self, Self]]:
        """
        Perform K-fold split such that all samples of a patient (identified by group_label)
        are in either train or validation, but never split across both.
        Uses StratifiedGroupKFold for splitting.
        :param folds: Number of folds.
        :param label: Column name for stratification labels.
        :param group_label: Column name for grouping (e.g., 'patient_id').
        :param random_state: Random seed.
        :return: List of tuples (train_dataset, val_dataset) for each fold.
        """
        y_cont = self.data_df[label].values

        # Map continuous values to class labels (0,1,2)
        mapping = {0.0: 0, 0.05: 0, 0.1: 0, 1.0: 1}
        try:
            y = np.array([mapping[val] for val in y_cont], dtype=int)
        except KeyError as e:
            raise ValueError(f"Unexpected label value for stratification: {e.args[0]}")

        groups = self.data_df[group_label].values

        sgkf = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        splits = sgkf.split(self.data_df.index, y=y, groups=groups)
        return [(
            self._create_new_dataset(self.data_df.loc[train_idx]),
            self._create_new_dataset(self.data_df.loc[val_idx])
        ) for train_idx, val_idx in splits]

    def reduce_dimensions(self, n_components: int, random_state: int, sampler: Literal[
        "FactoryAnalysis", "FastICA", "IncrementalPCA", "KernelPCA", "LatentDirichletAllocation",
        "MiniBatchDictionaryLearning", "MiniBatchNMF", "MiniBatchSparsePCA", "NMF", "PCA", "SparseCoder", "SparsePCA",
        "TruncatedSVD", "Isomap", "LLE", "MDS", "SpectralEmbedding", "TSNE", "LDA"]) -> (
            ClassNamePrefixFeaturesOutMixin, Self):
        """
        Reduce the dimensions of the point data using various sklearn decomposition or manifold learning methods.
        :param n_components: Number of components/dimensions to reduce to.
        :param random_state: Random seed.
        :param sampler: The name of the sklearn method to use.
        :return: Tuple of (fitted_method, new_dataset_with_reduced_dimensions).
        """
        df_data = self.data_df[self.data_columns()]
        df_label = self.data_df[self.label_column]

        methods = {
            "FactoryAnalysis": FactorAnalysis,
            "FastICA": FastICA,
            "IncrementalPCA": IncrementalPCA,
            "KernelPCA": KernelPCA,
            "LatentDirichletAllocation": LatentDirichletAllocation,
            "MiniBatchDictionaryLearning": MiniBatchDictionaryLearning,
            "MiniBatchNMF": MiniBatchNMF,
            "MiniBatchSparsePCA": MiniBatchSparsePCA,
            "NMF": NMF,
            "PCA": PCA,
            "SparseCoder": SparseCoder,
            "SparsePCA": SparsePCA,
            "TruncatedSVD": TruncatedSVD,
            "Isomap": Isomap,
            "LLE": LocallyLinearEmbedding,
            "MDS": MDS,
            "SpectralEmbedding": SpectralEmbedding,
            "TSNE": TSNE,
            "LDA": LinearDiscriminantAnalysis,
        }
        method_class = methods[sampler]
        method_params = {
            'n_components': n_components,
            'random_state': random_state,
        }
        params = inspect.getfullargspec(method_class)[0]
        method = methods[sampler](**{key: value for key, value in method_params.items() if key in params})
        reduced_df_data, reduced_df_label = method.fit_transform(df_data, df_label)
        return method, self._create_new_dataset(pd.concat([reduced_df_data, reduced_df_label], axis=1))

    def undersample(self, random_state: int, sampler: Literal[
        "ClusterCentroids", "RandomUnderSampler", "FunctionSampler", "NearMiss", "EditedNearestNeighbours",
        "RepeatedEditedNearestNeighbours", "AllKNN", "CondensedNearestNeighbour"] = "ClusterCentroids") -> Self:
        """
        Perform undersampling on the dataset to handle class imbalance.
        :param random_state: Random seed.
        :param sampler: The name of the imblearn undersampling method to use.
        :return: A new dataset instance with resampled data.
        """
        df_data = self.data_df[self.data_columns()]
        df_label = self.data_df[self.label_column]

        methods = {
            "ClusterCentroids": ClusterCentroids,
            "RandomUnderSampler": RandomUnderSampler,
            "FunctionSampler": FunctionSampler,
            "NearMiss": NearMiss,
            "EditedNearestNeighbours": EditedNearestNeighbours,
            "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
            "AllKNN": AllKNN,
            "CondensedNearestNeighbour": CondensedNearestNeighbour,
        }
        method = methods[sampler](random_state=random_state)
        resampled_df_data, resampled_df_label = method.fit_resample(df_data, df_label)

        return self._create_new_dataset(pd.concat([resampled_df_data, resampled_df_label], axis=1))

    def oversample(self, random_state: int, sampler: Literal[
        "SMOTE", "ADASYN", "RandomOverSampler", "KMeansSMOTE", "BorderlineSMOTE", "SMOTEN", "SMOTENC",
        "SVMSMOTE"] = "SMOTE") -> Self:
        """
        Perform oversampling on the dataset to handle class imbalance.
        :param random_state: Random seed.
        :param sampler: The name of the imblearn oversampling method to use.
        :return: A new dataset instance with resampled data.
        """
        df_data = self.data_df[self.data_columns()]
        df_label = self.data_df[self.label_column]

        methods = {
            "SMOTE": SMOTE,
            "ADASYN": ADASYN,
            "RandomOverSampler": RandomOverSampler,
            "KMeansSMOTE": KMeansSMOTE,
            "BorderlineSMOTE": BorderlineSMOTE,
            "SMOTEN": SMOTEN,
            "SMOTENC": SMOTENC,
            "SVMSMOTE": SVMSMOTE,
        }
        method = methods[sampler](random_state=random_state)
        resampled_df_data, resampled_df_label = method.fit_resample(df_data, df_label)

        return self._create_new_dataset(pd.concat([resampled_df_data, resampled_df_label], axis=1))
