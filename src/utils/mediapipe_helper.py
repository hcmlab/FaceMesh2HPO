import pathlib
from itertools import chain
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFile
from loguru import logger
from tqdm import tqdm


def extract_face_meshes(files: List[str], min_face_detection_confidence: float = 0.1):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        output_facial_transformation_matrixes=True,
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=min_face_detection_confidence)

    result = []
    with FaceLandmarker.create_from_options(options) as landmarker:
        for file in tqdm(files, desc='Face Mesh Extraction'):
            if file.endswith('.png') or file.endswith('.jpg'):
                ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow PIL to load truncated images
                image_pil = Image.open(file).convert("RGB")  # Open image using Pillow
                image_rgb = np.array(image_pil)  # Convert to NumPy array
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)
            else:
                image = mp.Image.create_from_file(file)
            try:
                face_landmarker_result = landmarker.detect(image)
                if len(face_landmarker_result.face_landmarks) > 0:
                    landmarks = face_landmarker_result.face_landmarks[0]
                    landmarks = [np.asarray([landmark.x, landmark.y, landmark.z]) for landmark in landmarks]
                    if np.all(~np.isnan(landmarks)):  # Check that no landmarks contain NaN values
                        result.append([pathlib.Path(file).stem] + list(chain.from_iterable(landmarks)))
                    else:
                        logger.warning(f'Could not detect face landmarks: {file}')
            except Exception as e:
                logger.warning(f'Failed to extract face landmarks: {e}')
    data = np.asarray(result)
    ids = data[:, 0]
    coordinates = data[:, 1:].reshape((len(data), 3, 478)).astype(np.float64)
    return ids, coordinates
