"""
File: temp_tracker.py
Author: Takao Kakegawa
Date: 2024
Description: Script to view frame-by-frame temperature cut reading for detected face.
             A horizontal cut is taken of the cropped face thermal frame from the nose.
"""

import ast
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.getcwd())
import json
import tomllib  

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from homography_alignment.homography import homographic_blend

# Mediaipipe facial landmark imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SenXor imports
from senxor.filters import RollingAverageFilter
from thermalfaceid.stark import STARKFilter

# modularised imports
from thermalfaceid.processing import process_thermal_frame
from thermalfaceid.utils import *
from thermalfaceid.inference import *


fig, ax = plt.subplots()
line, = ax.plot([], [])

rotation_map = {'90': cv.ROTATE_90_CLOCKWISE,
                '-90': cv.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv.ROTATE_180}

with open("config.toml", "rb") as f:
  config = tomllib.load(f)

if __name__ == "__main__":
  #### Create the argument parser
  default_params = config['tool']['model_params']
  
  parser = argparse.ArgumentParser(description='Argument Parser for Webcam ID and Rotation')
  parser.add_argument('-webcam_id', type=int, default=default_params['webcam_id'], help='Webcam ID if default detected webcam is not Logitech cam')
  parser.add_argument('-rotation', type=int, default=default_params['rotation'], help='Rotation for webcam if needed')
  parser.add_argument('-height_ratio', type=float, default=default_params['height_ratio'], help='minimum height ratio of frame for face to occupy')
  parser.add_argument('-face_confidence', type=int, default=default_params['face_confidence'], help='facial landmark detection confidence threshold')
  
  args = parser.parse_args()
  webcam_id = args.webcam_id
  min_height_ratio = args.height_ratio
  rotation = rotation_map.get(str(args.rotation), None)
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
  
  firstbenchmarkcoords = [14,49,279,4,              
                          62,104,65,52,51,64,       # BLUE SIDE EYEBROW
                          295,333,292,294,281,282   # GREEN SIDE EYEBROW
                          ]
  ####

  cv.namedWindow("Display")
  cv.namedWindow("Display2")

  empty_frame = np.zeros((120,120,3))

  while True:
    pred = None
    ret, rgbframe = cam.read()
    data, _ = mi48.read()
    if ret and data is not None:
      rgbimg = np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]
      rgbimg = homographic_blend(rgbimg.copy(), empty_frame, local_M_120120)[:113,6:108]    # HOMOGRAPHY STEP HERE TO PROJECT RGB ONTO THERMAL PLANE
      thermal_raw = process_thermal_frame(data, ncols, nrows, minav, maxav,
                                          minav2, maxav2, frame_filter)
      thermal_frame = thermal_raw[:113,26:128]    # thermal and projected rgb have same dimensions (113, 102)
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

        # Update histogram plot
        ax.clear()
        # xx = [thermalcrop[y_coords[i],x_coords[i]] for i in firstbenchmarkcoords]
        xx = thermalcrop.flatten().astype(np.float32)
        ax.hist(xx, bins=10, color='blue', alpha=0.7)
        ax.set_title(f"std: {np.std(xx)}")

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv.imshow("Display2", plot_img)

        for x,y in zip(x_coordsbig, y_coordsbig):
          thermalimgbig = cv.circle(thermalimgbig, (x,y), 1, (0,0,0), -1)   # verify that the points project correctly.

      else:
        rgbimgbig = cv.resize(rgbimg, dsize=None, fx=RSCALE, fy=RSCALE)
        thermalimgbig = cv.resize(thermalimg, dsize=None, fx=RSCALE, fy=RSCALE)
        cv.imshow("Display2", empty_frame)

    cv.imshow("Display", np.vstack((rgbimgbig, thermalimgbig)))

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
