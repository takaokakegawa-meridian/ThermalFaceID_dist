"""
File: thermalfaceid/processing.py
Author: Takao Kakegawa
Date: 2024
Description: Script with all visual/thermal image processing related functions
"""

from typing import Tuple
import numpy as np
import cv2 as cv

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision

from senxor.utils import data_to_frame, remap
from senxor.filters import RollingAverageFilter
from thermalfaceid.stark import STARKFilter


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


def homography_contours(img: np.ndarray, pctl: int, x1: int, y1: int, x2: int, y2: int,
                        show_cont=True, show_centroid=False, minPct=0.004) -> Tuple[np.ndarray, list, list]:
  """function that gets/draws contours on input image for automated homography alignment use.
  Args:
      img (np.ndarray): input image.
      pctl (int): percentile bound value [0-100]
      show_cont (bool, optional): draw contours on output image. Defaults to True.
      show_centroid (bool, optional): draw centroids on output image. Defaults to False.
      minPct (float, optional): min percent area size of entire frame for contour selection. Defaults to 0.004.
  Returns:
      Tuple[np.ndarray, list, list]: Tuple containing the input image, contours, and centroids.
  """
  ret = img.copy()
  frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  thresholdbound = np.percentile(frame.flatten(), pctl)
  _, binary = cv.threshold(frame, thresholdbound, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # find countour
  
  if len(contours) < 1:
    return ret, [], []
  
  minArea = minPct * frame.shape[1] * frame.shape[0]
  contours_f = []
  centroids = []

  for c in contours:
    if cv.contourArea(c) > minArea:
      extLeft = np.min(c[:, :, 0])
      extRight = np.max(c[:, :, 0])
      extTop = np.min(c[:,:,1])
      extBot = np.max(c[:, :, 1])

      if extLeft >= x1 and extRight <= x2 and extTop >= y1 and extBot <= y2:
        contours_f.append(c)
        try:
          M = cv.moments(c)
          centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
          # centroid = cv.KeyPoint(centroid[0], centroid[1], np.sqrt(4*M["m00"]/np.pi))
          centroids.append(centroid)
        except Exception as e:
          centroids.append(None)

  if show_cont and len(contours_f) > 0:
    ret = cv.drawContours(ret, contours_f, -1, (0,255,0), 1)

  if show_centroid:
    for centroid in centroids:
      if centroid is not None:
        ret = cv.circle(ret, centroid, 2, (0,0,255), -1)

  return ret, contours_f, centroids


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
