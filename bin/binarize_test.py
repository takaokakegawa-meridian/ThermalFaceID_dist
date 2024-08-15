"""
File: /bin/binarize_test.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to run to observe contour/centroids on thermal/visual independently
             to potentially match for homography calibration.
"""

### things to explore:
### preprocessing of thermal image, i.e. edge detection/binary thresholding

import ast
import argparse
import os
import sys
sys.path.append(os.getcwd())
import tomllib
from collections import defaultdict
import heapq
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2 as cv

# SenXor imports
from senxor.filters import RollingAverageFilter
from thermalfaceid.stark import STARKFilter
from senxor.utils import remap, get_default_outfile

# modularised imports
from thermalfaceid.processing import *
from thermalfaceid.utils import *
from homography_alignment.homography import *


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
  parser.add_argument('-rgb_pctl', type=int, default=85, help='RGB percentile threshold for binarizing')
  parser.add_argument('-th_pctl', type=int, default=92, help='Thermal percentile threshold for binarizing')
  args = parser.parse_args()
  webcam_id = args.webcam_id
  rotation = rotation_map.get(str(args.rotation), None)
  rgb_pctl = args.rgb_pctl
  th_pctl = args.th_pctl
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

  #### SIFT + OTHERS
  MIN_MATCH_COUNT = 4
  sift = cv.SIFT_create()   # Initiate SIFT detector
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv.FlannBasedMatcher(index_params, search_params)   # Initiate flann-based matcher
  RSCALE = 2
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
      thermalimg = np.dstack((thermal_frame, thermal_frame, thermal_frame))
      rgbimg = (np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]).astype(np.uint8)
      rgbimg = binarize_img(rgbimg, revert=True)

      # kp1, des1 = sift.detectAndCompute(thermal_frame, None)
      # kp2, des2 = sift.detectAndCompute(rgbimg, None)
      
      rgbimg = np.dstack((rgbimg, rgbimg, rgbimg))
      rgbimg, rgb_contours, rgb_centroids = homography_contours(rgbimg, rgb_pctl, show_centroid=False)
      thermalimg, th_contours, th_centroids = homography_contours(thermalimg, th_pctl, show_centroid=False, minPct=0.0005)
      
      if len(th_centroids) == 24 and len(rgb_centroids) == 24:

        # kp2, des2 = get_contour_keypoints_descriptors(kp2, des2, rgb_contours, rgb_centroids)
        # kp1, des1 = get_contour_keypoints_descriptors(kp1, des1, th_contours, th_centroids)
        rgb_centroids = sorted(rgb_centroids, key=lambda k: (k[1], k[0]))
        th_centroids = sorted(th_centroids, key=lambda k: (k[1], k[0]))

        # print(np.array(rgb_centroids))

        for rgb_c in rgb_centroids:
          cv.circle(rgbimg, rgb_c, 4, (0,0,255), -1)

        for th_c in th_centroids:
          cv.circle(thermalimg, th_c, 1, (0,0,255), -1)

        # for kp in kp2:
        #   if kp is not None:
        #     cv.circle(rgbimg, (int(kp.pt[0]), int(kp.pt[1])), 4, (0,0,255), -1)

        # for kp in kp1:
        #   if kp is not None:
        #     cv.circle(thermalimg, (int(kp.pt[0]), int(kp.pt[1])), 1, (0,0,255), -1)

      cv.rectangle(rgbimg, (10,30), (470,360), (255,0,0), 1)    # rectangle bounds for rgb points
      cv.rectangle(thermalimg, (2,8), (113,96), (255,0,0), 1)    # rectangle bounds for rgb points

      cv.imshow("Thermal", thermalimg)
      cv.imshow("Visual", rgbimg)
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
