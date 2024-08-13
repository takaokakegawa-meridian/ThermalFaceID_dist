"""
File: /bin/justthermallandmarks.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to run to observe MediaPipe facial landmarking on just thermal image.
"""

### things to explore:
### preprocessing of thermal image, i.e. edge detection/binary thresholding

import ast
import argparse
import joblib
import os
import sys
sys.path.append(os.getcwd())
import tomllib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2 as cv

# Mediaipipe facial landmark imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SenXor imports
from senxor.filters import RollingAverageFilter
from thermalfaceid.stark import STARKFilter
from senxor.utils import remap

# modularised imports
from thermalfaceid.processing import *
from thermalfaceid.utils import *


rotation_map = {'90': cv.ROTATE_90_CLOCKWISE,
                '-90': cv.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv.ROTATE_180}

with open("config.toml", "rb") as f:
  config = tomllib.load(f)

if __name__ == "__main__":
  #### Load in config default params
  default_params = config['tool']['model_params']
  
  #### Create the argument parser
  parser = argparse.ArgumentParser(description='Argument Parser for Webcam ID and Rotation')
  parser.add_argument('-webcam_id', type=int, default=default_params['webcam_id'], help='Webcam ID if default detected webcam is not Logitech cam')
  parser.add_argument('-rotation', type=int, default=default_params['rotation'], help='Rotation for webcam if needed')
  parser.add_argument('-height_ratio', type=float, default=default_params['height_ratio'], help='minimum height ratio of frame for face to occupy')
  parser.add_argument('-face_confidence', type=int, default=default_params['face_confidence'], help='facial landmark detection confidence threshold')
  parser.add_argument('-liveness_threshold', type=float, default=default_params['liveness_threshold'], help='liveness threshold')
  parser.add_argument('-heat_threshold', type=float, default=default_params['heat_threshold'], help='thermal face variation threshold')
  args = parser.parse_args()
  webcam_id = args.webcam_id
  min_height_ratio = args.height_ratio
  rotation = rotation_map.get(str(args.rotation), None)
  liveness_threshold = args.liveness_threshold
  heat_threshold = args.heat_threshold
  ####

#### Create FaceDetection and FaceLandmarker objects.
landmarkoptions = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='mediapipe_models/face_landmarker_v2_with_blendshapes.task'),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_presence_confidence=args.face_confidence,
            num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(landmarkoptions)
####

#### start visual webcam and SenXor thermal cam
cam = cv.VideoCapture(webcam_id)

mi48_params = config['tool']['mi48_params']
mi48_params['regwrite'] = [ast.literal_eval(i) for i in mi48_params['regwrite']]
mi48 = config_mi48(mi48_params)
ncols, nrows = mi48.fpa_shape
mi48.start(stream=True, with_header=True)
####

#### STARK PARAMETERS HERE:
minav = RollingAverageFilter(N=15)
maxav = RollingAverageFilter(N=8)
minav2 = RollingAverageFilter(N=15)
maxav2 = RollingAverageFilter(N=8)

stark_params = config['tool']['stark_params']
stark_params['lm_ks'] = ast.literal_eval(stark_params['lm_ks'])
frame_filter = STARKFilter(stark_params)
####

#### HOMOGRAPHY MATRIX HERE:

RSCALE = 2
####
#   winName = "Display"
#   cv.namedWindow(winName)
cv.namedWindow("Thermal")
cv.namedWindow("Visual")

# empty_frame_big = np.ones((113 * RSCALE,103 * RSCALE,3)) * 255
empty_frame_th = np.ones((113,103,3)) * 255
empty_frame_vi = np.ones((480,480,3)) * 255

while True:
  pred = None
  ret, rgbframe = cam.read()
  data, _ = mi48.read()
  if ret and data is not None:
    thermal_raw = process_thermal_frame(data, ncols, nrows, minav, maxav,
        minav2, maxav2, frame_filter)
    thermal_frame = remap(thermal_raw[:,20:-20][:,:]).astype(np.uint8)    # thermal and projected rgb have same dimensions (113, 102)
    thermalimg = np.dstack((thermal_frame, thermal_frame, thermal_frame))
    rgbimg = (np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]).astype(np.uint8)
    thermalimg, th_contours, th_centroids = homography_contours(thermalimg, 95, show_centroid=True)
    rgbimg, rgb_contours, rgb_centroids = homography_contours(rgbimg, 85, show_centroid=True)

    cv.imshow("Thermal", thermalimg)
    cv.imshow("Visual", rgbimg)
  #   thermalimg = cv_render(remap(thermal_frame), resize=thermal_frame.shape[::-1], display=False)
  #   bw_thermalimg = to_blackwhite(thermalimg.copy())
  #   bw_thermalimgbig = cv.resize(bw_thermalimg, dsize=None, fx=RSCALE, fy=RSCALE)
  #   thermalimgbig = cv.resize(thermalimg, dsize=None, fx=RSCALE, fy=RSCALE)
  #   cv.imshow(winName, np.vstack((bw_thermalimgbig, thermalimgbig)))
  else:
    cv.imshow("Thermal", empty_frame_th)
    cv.imshow("Visual", empty_frame_vi)

  key = cv.waitKey(1)
  if key == ord("q"):
    print("Quitting...")
    break
  elif key == ord("p"):
    print("Paused...")
    while True:
      key = cv.waitKey(1) & 0xFF
      if key == ord("r"):
        print("Resumed...")
        break

cv.destroyAllWindows()
