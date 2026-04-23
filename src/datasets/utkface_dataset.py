import glob
import os
import pathlib

import numpy as np
import pandas as pd
from loguru import logger

from src.datasets.base_dataset import BaseFaceMeshDataset
from src.utils.mediapipe_helper import extract_face_meshes


class UTKFaceFaceMeshDataset(BaseFaceMeshDataset):
    """
    Dataset class for the UTKFace dataset, focusing on face meshes and demographic labels.
    The UTKFace dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity.
    This class handles the extraction of face meshes from images and parsing of demographic data from filenames.

    @inproceedings{zhifei2017cvpr,
      title={Age Progression/Regression by Conditional Adversarial Autoencoder},
      author={Zhang, Zhifei, Song, Yang, and Qi, Hairong},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2017},
      organization={IEEE}
    }
    """

    def __init__(self, label_column: str, root_dir: str, data_file: str, data_df: pd.DataFrame = None,
                 reference_mesh: np.ndarray = None, dimensions: int = 3):
        """
        Initialize the UTKFaceFaceMeshDataset.
        :param label_column: Name of the column to use as the target label.
        :param root_dir: Path to the directory containing the UTKFace images.
        :param data_file: Path to a CSV file to save or load the processed data from.
        :param data_df: Optional pre-loaded DataFrame of the dataset.
        :param reference_mesh: Optional reference mesh for alignment.
        :param dimensions: Dimensionality of the point cloud (typically 3 for face meshes).
        """
        super().__init__(label_column, root_dir, data_file, data_df, reference_mesh, dimensions)
        self.data_df.dropna(inplace=True)

    def data_columns(self):
        """
        Get the names of the columns containing face mesh point coordinates.
        :return: List of column names (integers or strings representing numbers).
        """
        return [c for c in self.data_df.columns.tolist() if isinstance(c, int) or c.isdigit()]

    def label_columns(self):
        """
        Get the names of the columns containing demographic or identification labels.
        :return: List of label column names.
        """
        return [c for c in self.data_df.filter(regex=r'\D+', axis=1).columns.tolist()]

    def _create_new_dataset(self, new_data_df: pd.DataFrame):
        """
        Internal factory method to create a new instance of this dataset with a filtered DataFrame.
        :param new_data_df: The DataFrame to use for the new dataset instance.
        :return: A new UTKFaceFaceMeshDataset instance.
        """
        dataset = UTKFaceFaceMeshDataset(self.label_column, self.root_dir, "", new_data_df, self._reference_mesh,
                                         self.dimensions)
        dataset.mask_pts = self.mask_pts
        return dataset

    def _load_data(self, file_output: str, reference_mesh: np.ndarray = None):
        """
        Extract face meshes from images and parse metadata from filenames.
        Saves the resulting DataFrame to a CSV file.
        :param file_output: CSV file path to save the processed data.
        :param reference_mesh: Optional reference mesh for alignment (currently unused in this implementation).
        :return: Processed pandas DataFrame containing face mesh coordinates and labels.
        """
        img_files = glob.glob(os.path.join(self.root_dir, '*.jpg'))
        ids, coordinates = extract_face_meshes(img_files)
        data_df = pd.DataFrame(data=coordinates.reshape((len(coordinates), -1)), index=ids, dtype=np.float64)
        label_array = []
        for file in img_files:
            image_id = pathlib.Path(file).stem
            metadata = image_id.split('_')
            metadata_as_int = [int(i) if len(i) > 0 else -1 for i in metadata]
            if -1 in metadata_as_int or len(metadata_as_int) < 4:
                logger.warning(f'Dataset inconsistency found in image: {image_id}')
            label_array.append([image_id] + metadata_as_int)
        label_df = pd.DataFrame(data=label_array, columns=['image_id', 'age', 'gender', 'race', 'date_time'])
        data_df = pd.merge(label_df, data_df, left_on='image_id', right_index=True, how='left')

        gender_mapping = {0: 'male', 1: 'female', -1: 'unknown'}
        ethnicity_mapping = {0: 'European', 1: 'African', 2: 'Asian', 3: 'Others', 4: 'Others', -1: 'Unknown'}

        data_df.columns = data_df.columns.astype(str)
        data_df['gender'] = data_df['gender'].apply(lambda x: gender_mapping[x])
        data_df['ethnicity'] = data_df['race'].apply(lambda x: ethnicity_mapping[x])
        data_df['age'] = data_df['age'].apply(lambda x: float(x))
        data_df.to_csv(file_output)
        return data_df
