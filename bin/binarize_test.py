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

# modularised imports
from thermalfaceid.processing import process_thermal_frame, homography_contours
from thermalfaceid.utils import config_mi48
from homography_alignment.homography import binarize_img


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
  bound_params = config['tool']['homography_align_params']
  rgb_x1 = bound_params['rgb_x1']
  rgb_y1 = bound_params['rgb_y1']
  rgb_x2 = bound_params['rgb_x2']
  rgb_y2 = bound_params['rgb_y2']
  th_x1 = bound_params['th_x1']
  th_y1 = bound_params['th_y1']
  th_x2 = bound_params['th_x2']
  th_y2 = bound_params['th_y2']

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

  all_Ms = None
  M_count = 0

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
      rgbimg, rgb_contours, rgb_centroids = homography_contours(rgbimg, rgb_pctl, rgb_x1, rgb_y1,
                                                                rgb_x2, rgb_y2, show_centroid=False)
      thermalimg, th_contours, th_centroids = homography_contours(thermalimg, th_pctl, th_x1, th_y1,
                                                                  th_x2, th_y2, show_centroid=False, minPct=0.0005)
      
      if len(th_centroids) == 24 and len(rgb_centroids) == 24:

        # kp2, des2 = get_contour_keypoints_descriptors(kp2, des2, rgb_contours, rgb_centroids)
        # kp1, des1 = get_contour_keypoints_descriptors(kp1, des1, th_contours, th_centroids)
        rgb_centroids = sorted(rgb_centroids, key=lambda k: (k[1], k[0]))
        th_centroids = sorted(th_centroids, key=lambda k: (k[1], k[0]))

        M, mask = cv.findHomography(np.array(rgb_centroids), np.array(th_centroids), method=cv.RANSAC)
        if all_Ms is None:
          all_Ms = M
        else:
          all_Ms += M
        
        M_count += 1

        for rgb_c in rgb_centroids:
          cv.circle(rgbimg, rgb_c, 4, (0,0,255), -1)

        for th_c in th_centroids:
          cv.circle(thermalimg, th_c, 1, (0,0,255), -1)


      cv.rectangle(rgbimg, (rgb_x1,rgb_y1), (rgb_x2,rgb_y2), (255,0,0), 1)    # rectangle bounds for rgb points
      cv.rectangle(thermalimg, (th_x1,th_y1), (th_x2,th_y2), (255,0,0), 1)    # rectangle bounds for rgb points

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
  
  all_Ms /= M_count

  M_savepath = os.path.join("Depth_FCN_2/model_res", "automatic_M.json")
  with open(M_savepath, "w") as f:
    json.dump({"M": all_Ms.tolist()}, f)
  
  print(f"Saved calculated homography matrix into: {M_savepath}")

  cv.destroyAllWindows()
