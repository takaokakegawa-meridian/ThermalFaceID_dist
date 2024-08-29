"""
File: /bin/homography_val.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to run to validate and check the homography matrix calculated in bi
             binarize_test.py.
"""

import ast
import argparse
import json
import os
import sys
sys.path.append(os.getcwd())
import tomllib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2 as cv

# SenXor imports
from senxor.filters import RollingAverageFilter
from thermalfaceid.stark import STARKFilter
from senxor.utils import remap
from senxor.display import cv_render

# modularised imports
from thermalfaceid.processing import process_thermal_frame
from thermalfaceid.utils import config_mi48
from homography_alignment.homography import homographic_blend_alpha


rotation_map = {'90': cv.ROTATE_90_CLOCKWISE,
                '-90': cv.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv.ROTATE_180}

with open("config.toml", "rb") as f:
  config = tomllib.load(f)

with open(os.path.join("Depth_FCN_2/model_res", "automatic_M.json"), "r") as f:
  M = np.array(json.load(f)['M'])

if __name__ == "__main__":
  #### Load in config default params
  default_params = config['tool']['model_params']
  
  #### Create the argument parser
  parser = argparse.ArgumentParser(description='Argument Parser for Webcam ID and Rotation')
  parser.add_argument('-webcam_id', type=int, default=default_params['webcam_id'], help='Webcam ID if default detected webcam is not Logitech cam')
  parser.add_argument('-rotation', type=int, default=default_params['rotation'], help='Rotation for webcam if needed')
  parser.add_argument('-rgb_pctl', type=int, default=85, help='RGB percentile threshold for binarizing')
  parser.add_argument('-th_pctl', type=int, default=92, help='Thermal percentile threshold for binarizing')
  parser.add_argument('-th_cmap', type=str, default="jet", help='Thermal image colormap')
  args = parser.parse_args()
  webcam_id = args.webcam_id
  rotation = rotation_map.get(str(args.rotation), None)
  th_cmap = args.th_cmap
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

  ####
  cv.namedWindow("Thermal")
  cv.namedWindow("Visual")

  empty_frame_th = np.ones((113,103,3)) * 255
  empty_frame_vi = np.ones((480,480,3)) * 255
  ####

  while True:
    pred = None
    ret, rgbframe = cam.read()
    data, _ = mi48.read()
    if ret and data is not None:
      thermal_raw = process_thermal_frame(data, ncols, nrows, minav, maxav,
          minav2, maxav2, frame_filter)
      thermal_frame = remap(thermal_raw[:,20:-20][:,:]).astype(np.uint8)    # thermal and projected rgb have same dimensions (113, 102)
      rgbimg = (np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]).astype(np.uint8)

      thermalimg = cv_render(thermal_frame, resize=thermal_frame.shape[::-1],
                             colormap=th_cmap, display=False)

      homography_res = homographic_blend_alpha(rgbimg, thermalimg, M)

      cv.imshow("Thermal", thermalimg)
      cv.imshow("Visual", homography_res)
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
