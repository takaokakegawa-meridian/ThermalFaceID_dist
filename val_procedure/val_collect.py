"""
File: val_procedure/val_collect.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to run to observe plain CV window demonstrating Facial Anti-Spoofing with thermal data.
"""

import ast
import argparse
import os
import sys
sys.path.append(os.getcwd())
import json
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

# modularised imports
from utils import *
from homography_alignment.homography import homographic_blend
from thermalfaceid.processing import *
from thermalfaceid.inference import frame_inference_onlylandmarker
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
  parser.add_argument('-class_arg', type=int, default=1, help='Class. 1 is real, 0 is fake/spoof.')

  args = parser.parse_args()
  webcam_id = args.webcam_id
  min_height_ratio = args.height_ratio
  class_arg = args.class_arg
  rotation = rotation_map.get(str(args.rotation), None)
  if class_arg not in [0, 1]:
    print("Invalid class argument. Class should be 0 (fake) or 1 (real).")
    sys.exit(1)
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
  with open('homography_alignment/homographymatrix.json', "r") as f:
    local_M_120120 = json.load(f)
    local_M_120120 = np.array(local_M_120120['matrix']).astype(np.float64)
  
  RSCALE = 2
  ####

  #### forming validation data directory
  ret, rgbframe = cam.read()
  data, _ = mi48.read()
  if ret and data is not None:
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    target_dir = os.path.join(desktop_path, 'thermalfaceid_val')
    save_dir = os.path.join(target_dir, "real") if class_arg == 1 else os.path.join(target_dir, "fake")
    create_folders(target_dir)
  else:
    sys.exit("Connection issue with MI48 and/or visual camera.")
  ####

  winName = "Display"
  cv.namedWindow(winName)

  empty_frame = np.zeros((120,120,3))
  save_data = False

  while True:
    pred = None
    ret, rgbframe = cam.read()
    data, _ = mi48.read()
    if ret and data is not None:
      rgbimg = np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]
      rgbimg = homographic_blend(rgbimg.copy(), empty_frame, local_M_120120)[:,:]    # HOMOGRAPHY STEP HERE TO PROJECT RGB ONTO THERMAL PLANE
      thermal_raw = process_thermal_frame(data, ncols, nrows, minav, maxav,
                                          minav2, maxav2, frame_filter)
      thermal_frame = thermal_raw[:,20:-20][:,:]    # thermal and projected rgb have same dimensions (113, 102)
      rgbimg, thermalimg, landmarks = frame_inference_onlylandmarker(rgbimg, thermal_frame, landmarker, min_height_ratio, 'jet')  # inference here

      if landmarks is not None:
        yscale, xscale = rgbimg.shape[:2]
        y_coords = (landmarks[1][:,1] * yscale).astype(int)
        x_coords = (landmarks[1][:,0] * xscale).astype(int)

        rgbimgbig = cv.resize(rgbimg, dsize=None, fx=RSCALE, fy=RSCALE)
        thermalimgbig = cv.resize(thermalimg, dsize=None, fx=RSCALE, fy=RSCALE)
        yscalebig, xscalebig = rgbimgbig.shape[:2]
        rgbimgbig = draw_landmarks_on_image(rgbimgbig, landmarks[0])


        y_coordsbig = (landmarks[1][:,1] * yscalebig).astype(int)
        x_coordsbig = (landmarks[1][:,0] * xscalebig).astype(int)

        xminbig, xmaxbig = max(np.min(x_coordsbig) - 1, 0), np.max(x_coordsbig) + 2
        yminbig, ymaxbig = max(np.min(y_coordsbig) - 1, 0), np.max(y_coordsbig) + 2
        xmin, xmax = max(np.min(x_coords) - 1, 0), np.max(x_coords) + 2
        ymin, ymax = max(np.min(y_coords) - 1, 0), np.max(y_coords) + 2

        y_coords -= ymin
        x_coords -= xmin
        
        # getting the crops:
        rgbcrop = rgbimg[ymin:ymax,xmin:xmax]
        thermalcrop = thermal_frame[ymin:ymax,xmin:xmax]

        for x,y in zip(x_coordsbig, y_coordsbig):
          thermalimgbig = cv.circle(thermalimgbig, (x,y), 1, (0,0,0), -1)   # verify that the points project correctly.
        
        rgbimgbig = cv.rectangle(rgbimgbig, (xminbig, yminbig), (xmaxbig, ymaxbig), (255, 0, 0), 1)

        if save_data:   # saving data: saving thermalcrop, rgbcrop and landmarkcoords
          save_val_data(save_dir, x_coords, y_coords, rgbcrop, thermalcrop)

      else:
        rgbimgbig = cv.resize(rgbimg, dsize=None, fx=RSCALE, fy=RSCALE)
        thermalimgbig = cv.resize(thermalimg, dsize=None, fx=RSCALE, fy=RSCALE)

    cv.imshow(winName, np.vstack((rgbimgbig, thermalimgbig)))

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
    elif key == ord("m"):
      save_data = True
      print("Save images ENABLED")
    elif key == ord("n"):
      save_data = False
      print("Save images DISABLED")


  mi48.stop()
  cv.destroyAllWindows()
