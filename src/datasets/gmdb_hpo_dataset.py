import os

import numpy as np
import pandas as pd
import tqdm
from loguru import logger

from src.datasets.gmdb_dataset import GMDBFaceMeshDataset


class GMDBFaceMeshHPODataset(GMDBFaceMeshDataset):
    """
    Extended dataset class for GMDB face mesh data incorporating Human Phenotype Ontology (HPO) annotations.
    This class handles merging standard GMDB features with manually annotated HPO terms and filtering
    the dataset based on specific annotation types.
    """
    ALL = 0
    GMDB_WITH_FACEMESH_ANNOTATIONS = 1
    GMDB_WITH_HPO_ANNOTATIONS = 2
    GMDB_WITH_OVERLAPPING_ANNOTATIONS = 3
    GMDB_WITH_OVERLAPPING_ANNOTATIONS_MEDIAPIPE_FACE_MESHES = 4

    def __init__(self, label_column: str, root_dir: str, gmdb_data_file: str, extended_data_file: str,
                 data_df: pd.DataFrame = None,
                 reference_mesh: np.ndarray = None, dimensions: int = 3):
        """
        Initialize the GMDBFaceMeshHPODataset.
        :param label_column: Name of the column containing the labels.
        :param root_dir: Root directory of the dataset.
        :param gmdb_data_file: Path to the base GMDB data CSV file.
        :param extended_data_file: Path to the CSV file where the merged HPO data will be saved.
        :param data_df: Optional pre-loaded DataFrame.
        :param reference_mesh: Optional reference mesh for Procrustes alignment.
        :param dimensions: Number of dimensions for the points (e.g., 2 or 3).
        """
        self.gmdb_data_file = gmdb_data_file
        self.extended_data_file = extended_data_file
        super().__init__(label_column, root_dir, extended_data_file, data_df, reference_mesh, dimensions)

    def update_by_type(self, new_type):
        """
        Filters the internal DataFrame based on the specified annotation type.
        :param new_type: One of the class constants (e.g., GMDB_WITH_HPO_ANNOTATIONS).
        """
        if new_type == self.GMDB_WITH_FACEMESH_ANNOTATIONS:
            mask = (self.data_df['present_features'].notna() | self.data_df['absent_features'].notna()) & self.data_df[
                'present_hpo_terms'].isna() & self.data_df['absent_hpo_terms'].isna() & self.data_df['manual_0'].notna()
            self.data_df = self.data_df[mask]
        elif new_type == self.GMDB_WITH_HPO_ANNOTATIONS:
            mask = (self.data_df['present_hpo_terms'].notna() | self.data_df['absent_hpo_terms'].notna())
            self.data_df = self.data_df[mask]
        elif new_type == self.GMDB_WITH_OVERLAPPING_ANNOTATIONS:
            mask = (self.data_df['present_hpo_terms'].notna() | self.data_df['absent_hpo_terms'].notna()) & \
                   self.data_df['manual_0'].notna()
            self.data_df = self.data_df[mask]
        elif new_type == self.GMDB_WITH_OVERLAPPING_ANNOTATIONS_MEDIAPIPE_FACE_MESHES:
            mask = (self.data_df['present_hpo_terms'].notna() | self.data_df['absent_hpo_terms'].notna())
            self.data_df = self.data_df[mask]

    def _create_new_dataset(self, new_data_df: pd.DataFrame):
        """
        Create a new instance of GMDBFaceMeshHPODataset with a different DataFrame.
        :param new_data_df: The new DataFrame to use.
        :return: A new instance of GMDBFaceMeshHPODataset.
        """
        dataset = GMDBFaceMeshHPODataset(self.label_column, self.root_dir, self.gmdb_data_file, self.extended_data_file,
                                         new_data_df,
                                         self._reference_mesh, self.dimensions)
        dataset.mask_pts = self.mask_pts
        return dataset

    def _load_data(self, file: str, reference_mesh: np.ndarray = None):
        """
        Load and merge GMDB data with manual HPO annotations.
        Merges base GMDB features with HPO terms from 'Face2HPO-181120251320.csv',
        handling duplicate entries and synchronizing feature sets.
        :param file: Path to save the processed CSV file.
        :param reference_mesh: Optional reference mesh for alignment during loading.
        :return: Processed pandas DataFrame containing merged annotations.
        """
        hpo_data_file = os.path.join(self.root_dir, 'Face2HPO-181120251320.csv')
        hpo_df = pd.read_csv(hpo_data_file)
        logger.debug(f'Found {len(hpo_df)} HPO entries in the manually annotated HPO dataset.')
        columns = ['image_id', 'present_hpo_terms', 'absent_hpo_terms']
        redundancy_mask = hpo_df['image_id'].duplicated(keep='first')
        duplicates_df = hpo_df[redundancy_mask]
        logger.debug(f'Found {len(duplicates_df)} duplicate entries in the manually annotated HPO dataset.')
        for idx, row in tqdm.tqdm(duplicates_df.iterrows(), desc='Merging Duplicates', total=len(duplicates_df)):
            duplicate_rows = hpo_df[hpo_df['image_id'] == row['image_id']]
            for column in columns[1:]:
                merged_values = set()
                for row_idx in range(len(duplicate_rows)):
                    hpo_entries = duplicate_rows.iloc[row_idx][column]
                    annotated_set = hpo_entries.split(' ') if isinstance(hpo_entries, str) else []
                    merged_values.update(annotated_set)
                hpo_list = ' '.join(merged_values)
                hpo_df.loc[hpo_df['image_id'] == row['image_id'], column] = hpo_list

        hpo_df.drop_duplicates(subset=['image_id'], inplace=True, keep='first')
        hpo_df.drop(columns=['user_id', 'username', 'patient_id', 'omim_id'], inplace=True)
        hpo_df = hpo_df[columns]
        logger.debug(f'After merging duplicates, we have {len(hpo_df)} data samples to work with.')

        gmdb_df = pd.read_csv(self.gmdb_data_file)
        df = pd.merge(gmdb_df, hpo_df, on='image_id', how='left')

        def merge_features(row):
            """
            Internal helper to merge HPO terms into the standard feature columns.
            Ensures consistency between 'present_features' and 'present_hpo_terms'.
            :param row: A row of the merged DataFrame.
            :return: The updated row with 'present_features_merged' and 'absent_features_merged'.
            """

            # Helper to split strings safely into sets
            def split_to_set(value, sep):
                if pd.isna(value):
                    return set()
                return set(s.strip() for s in str(value).split(sep) if s.strip())

            # Parse present_features and absent_features (semicolon separated)
            present_feats = split_to_set(row['present_features'], ';')
            absent_feats = split_to_set(row['absent_features'], ';')

            # Parse present_hpo_terms and absent_hpo_terms (space separated)
            present_hpo = split_to_set(row['present_hpo_terms'], ' ')
            absent_hpo = split_to_set(row['absent_hpo_terms'], ' ')

            # Add present_hpo terms to present_features if not already in absent_features
            new_present = {term for term in present_hpo if term not in absent_feats}

            # Add absent_hpo terms to absent_features if not already in present_features
            new_absent = {term for term in absent_hpo if term not in present_feats}

            # Merge and convert back to semicolon-separated strings
            present_feats.update(new_present)
            absent_feats.update(new_absent)

            row['present_features_merged'] = ';'.join(sorted(present_feats)) if present_feats else np.nan
            row['absent_features_merged'] = ';'.join(sorted(absent_feats)) if absent_feats else np.nan

            return row

        # Assuming df is the merged dataframe with the target columns:
        df_merged = df.apply(merge_features, axis=1)
        df_all = df_merged.copy()
        df_all.drop(columns=['Unnamed: 0'], inplace=True)
        # df_all.drop(df_all.filter(regex='HP:*').columns, axis=1, inplace=True)
        df_all.to_csv(file)
        return df_all
