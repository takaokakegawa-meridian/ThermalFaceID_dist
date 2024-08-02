"""
File: processing.py
Author: Takao Kakegawa
Date: 2024
Description: Script with all visual/thermal image processing related functions
"""

import numpy as np
import cv2 as cv

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision

from senxor.utils import data_to_frame
from senxor.filters import RollingAverageFilter
from stark import STARKFilter


def convert_bgr_cnn_input(bgr_frame: np.ndarray) -> np.ndarray:
  """convert bgr CV image object into HSV + YCrCb representation for colomap transformation
  Args:
    bgr_frame (np.ndarray): 3D numpy array representing visual frame in BGR format.
  Returns:
    np.ndarray: input image transformed into HSV + YCrCb representation in BGR format.
  """
  frame_copy = bgr_frame.copy()
  hsv = cv.cvtColor(frame_copy, cv.COLOR_BGR2HSV)
  ycrcb = cv.cvtColor(frame_copy, cv.COLOR_BGR2YCrCb)

  return hsv + ycrcb


def process_thermal_frame(data: np.ndarray, ncols: int, nrows: int, minav: RollingAverageFilter,
                          maxav: RollingAverageFilter, minav2: RollingAverageFilter,
                          maxav2: RollingAverageFilter, frame_filter: STARKFilter) -> np.ndarray:
  """convert raw data output from mi48 into thermal frame, cleaned with applied filters.
  Args:
    data (np.ndarray): numpy array that is the raw data output from mi48. Should be shape (nrows*ncols,)
    ncols (int): number of cols in mi48 output.
    nrows (int): number of rows in mi48 output.
    minav (RollingAverageFilter): filter for first clipping filter step using frame's min val
    maxav (RollingAverageFilter): filter for first clipping filter step using frame's max val
    minav2 (RollingAverageFilter): filter for second clipping filter step using frame's min val
    maxav2 (RollingAverageFilter): filter for second clipping filter step using frame's max val
    frame_filter (STARKFilter): tuned STARK filter that we apply after clipping
  Returns:
    np.ndarray: 2D numpy array representing the clean/filtered thermal frame.
  """
  thermal_frame = data_to_frame(data, (ncols, nrows), hflip=False)
  sorted_frame = np.sort(thermal_frame.flatten())
  min_temp1= minav(np.median(sorted_frame[:16]))
  max_temp1= maxav(np.median(sorted_frame[-5:]))
  thermal_frame = frame_filter(np.clip(thermal_frame, min_temp1, max_temp1))
  sorted_frame = np.sort(thermal_frame.flatten())
  min_temp2 = minav2(np.median(sorted_frame[:9]))
  max_temp2= maxav2(np.median(sorted_frame[-5:]))
  thermal_frame = np.clip(thermal_frame, min_temp2, max_temp2)
  return thermal_frame


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: vision.FaceLandmarkerResult) -> np.ndarray:
  """draw detected landmark entities on input image.
  Args:
    rgb_image (np.ndarray): The input image as a 3-channel numpy array in BGR format.
    detection_result (vision.FaceLandmarkerResult): The vision.FaceLandmarkerResult object containing information on the detected landmarks.
  Returns:
    np.ndarray: Image with landmarks and facial mask annotated onto input image.
  """

  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

  # Draw the face landmarks.
  face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  face_landmarks_proto.landmark.extend([
  landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
  ])

  solutions.drawing_utils.draw_landmarks(
  image=annotated_image,
  landmark_list=face_landmarks_proto,
  connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
  landmark_drawing_spec=None,
  connection_drawing_spec=mp.solutions.drawing_styles
  .get_default_face_mesh_tesselation_style())

  solutions.drawing_utils.draw_landmarks(
  image=annotated_image,
  landmark_list=face_landmarks_proto,
  connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
  landmark_drawing_spec=None,
  connection_drawing_spec=mp.solutions.drawing_styles
  .get_default_face_mesh_contours_style())

  solutions.drawing_utils.draw_landmarks(
  image=annotated_image,
  landmark_list=face_landmarks_proto,
  connections=mp.solutions.face_mesh.FACEMESH_IRISES,
  landmark_drawing_spec=None,
  connection_drawing_spec=mp.solutions.drawing_styles
  .get_default_face_mesh_iris_connections_style())

  return annotated_image
