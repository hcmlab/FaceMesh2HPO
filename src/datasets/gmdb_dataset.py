import glob
import os
import warnings

import numpy as np
import pandas as pd
import tqdm

from src.datasets.base_dataset import BaseFaceMeshDataset
from src.hpo_tree.hpo_term import HumanPhenotypeTerm
from src.utils.mediapipe_helper import extract_face_meshes

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class GMDBFaceMeshDataset(BaseFaceMeshDataset):
    """
    Dataset class for loading and processing the GMDB face mesh data.
    GMDB (Global Medical Discovery Database) dataset includes images, metadata, and face meshes.
    This class handles extraction of face meshes, mapping metadata (age, ethnicity), and HPO table creation.
    """
    def __init__(self, label_column: str, root_dir: str, data_file: str, data_df: pd.DataFrame = None,
                 reference_mesh: np.ndarray = None, dimensions: int = 3):
        """
        Initialize the GMDBFaceMeshDataset.
        :param label_column: Name of the column containing the labels.
        :param root_dir: Root directory of the dataset.
        :param data_file: Path to the CSV file where the processed data is (or will be) saved.
        :param data_df: Optional pre-loaded DataFrame.
        :param reference_mesh: Optional reference mesh for Procrustes alignment.
        :param dimensions: Number of dimensions for the points (e.g., 2 or 3).
        """
        super().__init__(label_column, root_dir, data_file, data_df, reference_mesh, dimensions)

    def data_columns(self):
        """
        Identify columns containing face mesh coordinate data.
        Assumes coordinate columns are either integers or digit-strings.
        :return: List of column names (strings or ints).
        """
        return [c for c in self.data_df.columns.tolist() if isinstance(c, int) or c.isdigit()]

    def label_columns(self):
        """
        Identify columns containing HPO labels.
        Labels are expected to start with the 'HP:' prefix.
        :return: List of column names starting with 'HP:'.
        """
        return [c for c in self.data_df.filter(regex=r'HP:*', axis=1).columns.tolist()]

    def _create_new_dataset(self, new_data_df: pd.DataFrame):
        """
        Create a new instance of GMDBFaceMeshDataset with a different DataFrame.
        Used for splitting or sampling the dataset.
        :param new_data_df: The new DataFrame to use.
        :return: A new instance of GMDBFaceMeshDataset.
        """
        dataset = GMDBFaceMeshDataset(self.label_column, self.root_dir, "", new_data_df, self._reference_mesh,
                                      self.dimensions)
        dataset.mask_pts = self.mask_pts
        return dataset

    def _load_data(self, file: str, reference_mesh: np.ndarray = None):
        """
        Load raw GMDB data from images and metadata files, then process them into a single DataFrame.
        Extracts face meshes from images, merges with metadata, and saves to a CSV.
        :param file: Path to save the processed CSV file.
        :param reference_mesh: Optional reference mesh for alignment during loading.
        :return: Processed pandas DataFrame.
        """
        img_files = glob.glob(os.path.join(self.root_dir, 'gmdb_images', '*.jpg'))
        meta_df = pd.read_csv(glob.glob(os.path.join(self.root_dir, 'gmdb_metadata', 'image_metadata_*.tsv'))[0],
                              delimiter='\t', low_memory=False)
        ids, coordinates = extract_face_meshes(img_files)
        ids = [int(id) for id in ids]
        data_df = pd.DataFrame(data=coordinates.reshape((len(coordinates), -1)), index=ids, dtype=np.float64)
        tmp_index = meta_df.index
        data_df = pd.merge(meta_df, data_df, left_on='image_id', right_index=True, how='left')
        data_df.index = tmp_index
        data_df.columns = data_df.columns.astype(str)
        data_df['ethnicity'] = data_df['ethnicity_category']
        data_df['age'] = data_df['age_year']
        data_df.to_csv(file)
        return data_df

    def create_hpo_table(self, hpo: HumanPhenotypeTerm, present_feature: str = 'present_features',
                         absent_feature: str = 'absent_features') -> None:
        """
        Creates an HPO table by propagating phenotypic features through the HPO ontology.
        Labels are added as new columns to the internal DataFrame: 1 for present, 0 for absent, -1 for unknown.
        It uses the following logic:
        - If a feature is present, all its predecessors (parents) are also present (set to 1).
        - If a feature is absent, all its successors (children) are also absent (set to 0).
        - If all children of a feature are absent, the feature itself is inferred as absent (set to 0).
        - Otherwise, the status is unknown (set to -1).
        :param hpo: The root HPO term for the ontology tree.
        :param present_feature: Column name in data_df containing semicolon-separated present HPO IDs.
        :param absent_feature: Column name in data_df containing semicolon-separated absent HPO IDs.
        """
        all_hpo = {h.id: h for h in hpo.all_successors(with_self=True)}
        image_id_tag = 'image_id'
        all_hpo_values = {image_id_tag: []}
        all_hpo_values.update({hpo_id: [] for hpo_id in all_hpo.keys()})

        for index, row in tqdm.tqdm(self.data_df.iterrows(), total=self.data_df.shape[0], desc='Creating HPO table'):
            all_hpo_values[image_id_tag].append(row[image_id_tag])

            # Parse present/absent features once
            present_ids = (
                    set(row[present_feature].split(';') if isinstance(row[present_feature], str)
                        else [])
                    & set(all_hpo)  # Only known HPOs
            )

            absent_ids = (
                    set(row[absent_feature].split(';') if isinstance(row[absent_feature], str)
                        else [])
                    & set(all_hpo)
            )

            # Propagate present → all predecessors (including self)
            present_with_preds = set()
            for hpo_id in present_ids:
                present_with_preds.update([p.id for p in all_hpo[hpo_id].predecessors(with_self=True)])

            # Propagate absent → all successors (including self)
            absent_with_succs = set()
            for hpo_id in absent_ids:
                absent_with_succs.update([s.id for s in all_hpo[hpo_id].all_successors(with_self=True)])

            # Infer absent predecessors
            for hpo_id in all_hpo:
                if hpo_id in present_with_preds:
                    all_hpo_values[hpo_id].append(1)
                    continue

                if hpo_id in absent_with_succs:
                    all_hpo_values[hpo_id].append(0)
                    continue

                # Check if ALL successors of this HPO are absent
                all_successors_absent = True
                for succ in all_hpo[hpo_id].all_successors():
                    if succ.id not in absent_with_succs:
                        all_successors_absent = False
                        break

                if all_successors_absent:
                    all_hpo_values[hpo_id].append(0)  # Infer absent
                else:
                    all_hpo_values[hpo_id].append(-1)  # Unknown

        labels_df = pd.DataFrame(data=all_hpo_values, columns=list(all_hpo_values.keys()))
        tmp_index = self.data_df.index
        self.data_df = pd.merge(self.data_df, labels_df, on='image_id', how='left')
        self.data_df.index = tmp_index
