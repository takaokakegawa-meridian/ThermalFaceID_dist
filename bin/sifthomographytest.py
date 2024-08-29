"""
File: /bin/sifthomographytest.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to try SIFT for homography estimation
"""

### things to explore:
### preprocessing of thermal image, i.e. edge detection/binary thresholding

import ast
import argparse
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
from senxor.display import cv_render
from thermalfaceid.stark import STARKFilter
from senxor.utils import remap, get_default_outfile

# modularised imports
from thermalfaceid.processing import *
from thermalfaceid.utils import *
from homography_alignment.homography import homographic_blend, homographic_blend_alpha


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

#### HOMOGRAPHY ESTIMATION PARAMETERS HERE:
MIN_MATCH_COUNT = 4
sift = cv.SIFT_create(nfeatures=24)   # Initiate SIFT detector
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)   # Initiate flann-based matcher
RSCALE = 2
####
#   winName = "Display"
#   cv.namedWindow(winName)
cv.namedWindow("Thermal")
cv.namedWindow("Visual")

# empty_frame_big = np.ones((113 * RSCALE,103 * RSCALE,3)) * 255
empty_frame_th = np.ones((113,103,3)) * 255
empty_frame_vi = np.ones((480,480,3)) * 255

found_M = False
M_res = None

while True:
  pred = None
  ret, rgbframe = cam.read()
  data, _ = mi48.read()
  if ret and data is not None:
    thermal_raw = process_thermal_frame(data, ncols, nrows, minav, maxav,
        minav2, maxav2, frame_filter)
    thermal_frame = remap(thermal_raw[:,20:-20][:,:]).astype(np.uint8)    # thermal and projected rgb have same dimensions (113, 102)


    thermalimg = np.dstack((thermal_frame, thermal_frame, thermal_frame))
    displaythermal = thermalimg.copy()
    rgbimg = (np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]).astype(np.uint8)

    ##### try to binarize rgbimg for preprocessing. try to improve way of getting keypoints.
    ##### https://medium.com/analytics-vidhya/image-simplification-through-binarization-in-opencv-1292d91cae12
    displayrgb = rgbimg.copy()

    if not found_M:
      rgbgray = cv.cvtColor(rgbimg, cv.COLOR_BGR2GRAY)
      kp2, des2 = sift.detectAndCompute(rgbgray,None)
      for point in cv.KeyPoint_convert(kp2):
        displayrgb = cv.circle(displayrgb, (int(point[0]), int(point[1])), 2, (0,0,255), -1)

      kp1, des1 = sift.detectAndCompute(thermal_frame, None)
      for point in cv.KeyPoint_convert(kp1):
        displaythermal = cv.circle(displaythermal, (int(point[0]), int(point[1])), 2, (0,0,255), -1)

      matches = flann.knnMatch(des1,des2,k=2)
      good = []
      for m,n in matches:
        if m.distance < 0.7*n.distance:
          good.append(m)

      if len(good) >= MIN_MATCH_COUNT:
        print("enough points for homography")
        src_pts = [kp1[m.queryIdx].pt for m in good]
        dst_pts = [kp2[m.trainIdx].pt for m in good]
        src_pts = [(int(i[0]),int(i[1])) for i in src_pts]
        dst_pts = [(int(i[0]),int(i[1])) for i in dst_pts]

        print(f"len: {len(src_pts)}", src_pts)
        print(f"len: {len(dst_pts)}", dst_pts)

        for src_pt in src_pts:
          thermalimg = cv.circle(thermalimg, src_pt, 2, (0,0,255), -1)
        for dst_pt in dst_pts:
          rgbimg = cv.circle(rgbimg, dst_pt, 2, (0,0,255), -1)

        # img3 = cv.drawMatchesKnn(thermalimg,kp1,rgbimg,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        savefile = get_default_outfile()
        cv.imwrite('bin/bin_imgs/thermal_'+savefile, thermalimg)
        cv.imwrite('bin/bin_imgs/rgb_'+savefile, rgbimg)
        sys.exit(0)
        
    #     dst_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    #     src_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    #     print(f"src_pts.shape: {src_pts.shape}, dst_pts.shape: {dst_pts.shape}")
    #     M_res, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #     if M_res is not None:
    #       found_M = True

    # else:
    #   rgbimg = homographic_blend_alpha(rgbimg, displaythermal, M_res, 0.1)

    cv.imshow("Thermal", displaythermal)
    cv.imshow("Visual", displayrgb)

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
