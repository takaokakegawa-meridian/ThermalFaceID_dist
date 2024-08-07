"""
File: thermalfaceid/inference.py
Author: Takao Kakegawa
Date: 2024
Description: Script with all inference-related functions to compartmentalize end-to-end frame 
             inference within one function.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.pipeline import Pipeline
import torch
import torch.nn.functional as F
import mediapipe as mp

from senxor.utils import remap
from senxor.display import cv_render

from Depth_FCN_2.FCN import DepthBasedFCN
from thermalfaceid.processing import *


def FC2_predict(thermalcrop: np.ndarray, rgbimg: np.ndarray, landmarkcoords: tuple,
                NN_model: DepthBasedFCN, SVMclf: Pipeline) -> bool:
  """Execute FCN Depth Based model end-to-end inference process for current frame.
  Args:
    thermalcrop (np.ndarray): 2D thermal frame cropped to face bounds.
    rgbimg (np.ndarray): 3D numpy array representing face image in BGR format.
    landmarkcoords (tuple): tuple of x/y coordinate lists for the facial landmark coordinates
    NN_model (DepthBasedFCN): FCN Depth based model.
    SVMclf (Pipeline): sklearn pipeline that applied SVM to NN_model output for binary output
  Returns:
    bool: True/False output is face is real or not.
  """
  x_coords, y_coords = landmarkcoords
  modelinput = convert_bgr_cnn_input(rgbimg)
  inputimg = torch.from_numpy(modelinput).permute(2, 0, 1).float()
  output = NN_model(inputimg).unsqueeze(dim=0)
  output_resized = F.interpolate(output, size=thermalcrop.shape,
                                  mode='bicubic').squeeze()
  output_vals = output_resized[y_coords, x_coords].detach().numpy()
  thactual = thermalcrop[y_coords, x_coords]
  svminput = (output_vals - thactual) ** 2
  pred = SVMclf.predict(svminput.reshape(1,-1))[0]
  return True if pred == 1 else False


###### INFERENCE DRAWING BOUNDS ONLY WITH LANDMARKER
def frame_inference_onlylandmarker(rgbimg: np.ndarray, thermal_frame: np.ndarray, landmarker: vision.FaceLandmarker,
                                   min_height_ratio: float, colormap: str = 'bone') -> \
                                   Tuple[np.ndarray, np.ndarray, Optional[Tuple[vision.FaceLandmarkerResult,np.ndarray]]]:
  """ Perform facial landmark detection on a given RGB image and thermal frame using a specified FaceLandmarker.
  Args:
    rgbimg (np.ndarray): The RGB image for facial landmark detection.
    thermal_frame (np.ndarray): The thermal frame for rendering.
    landmarker (vision.FaceLandmarker): The FaceLandmarker object for detecting facial landmarks.
    min_height_ratio (float): The minimum height ratio used for validation.
    colormap (str, optional): The colormap for rendering thermal image (default is 'bone').
  Returns:
    tuple[np.ndarray, np.ndarray, Optional[Tuple[vision.FaceLandmarkerResult, np.ndarray]]]: 
    A tuple containing the RGB image, thermal image, and facial landmark results if successful, or None.
  """
  thermalimg = cv_render(remap(thermal_frame), resize=(thermal_frame.shape[1],thermal_frame.shape[0]),
                         colormap=colormap, display=False)
  try:
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbimg.copy())
    landmarkresult = landmarker.detect(img)  # get facial landmarks on cropped face image
    landmark_list = landmarkresult.face_landmarks
    if len(landmark_list) > 0:
      x_coords = np.array([[landmark.x, landmark.y] for landmark in landmark_list[0] if 0 <= landmark.x <= 1. and 0 <= landmark.y <= 1.])
      if len(x_coords) == 478 and (np.max(x_coords[:, 1]) - np.min(x_coords[:, 1])) > min_height_ratio:
        return rgbimg, thermalimg, (landmarkresult, x_coords)
  except IndexError:
    pass

  return rgbimg, thermalimg, None
###### INFERENCE DRAWING BOUNDS ONLY WITH LANDMARKER
