"""
File: main.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to run to observe plain CV window demonstrating Facial Anti-Spoofing with thermal data.
"""

import argparse
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2 as cv
import torch
from Depth_FCN_2.FCN import DepthBasedFCN      # FCN2 model imports
from homography_alignment.homography import homographic_blend

# Mediaipipe facial landmark imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SenXor imports
from senxor.filters import RollingAverageFilter
from stark import STARKFilter


# modularised imports
from processing import *
from utils import *
from inference import *


rotation_map = {'90': cv.ROTATE_90_CLOCKWISE,
                '-90': cv.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv.ROTATE_180}


if __name__ == "__main__":
  #### Create the argument parser
  parser = argparse.ArgumentParser(description='Argument Parser for Webcam ID and Rotation')
  parser.add_argument('-webcam_id', type=int, default=0, help='Webcam ID if default detected webcam is not Logitech cam')
  parser.add_argument('-rotation', type=int, default=90, help='Rotation for webcam if needed')
  parser.add_argument('-height_ratio', type=float, default=0.75, help='minimum height ratio of frame for face to occupy')
  parser.add_argument('-face_confidence', type=int, default=0.7, help='facial landmark detection confidence threshold')
  parser.add_argument('-liveness_threshold', type=float, default=0.04, help='liveness threshold')
  parser.add_argument('-heat_threshold', type=float, default=1.5, help='thermal face variation threshold')
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

  #### Load both Patch-FCN model and SVM classifiers
  weight_root = "Depth_FCN_2/model_res"
  model = DepthBasedFCN(3)
  model.load_state_dict(torch.load(os.path.join(weight_root,"best_weights.pt"),
                        map_location=torch.device('cpu')))
  model.eval()
  SVMclf = joblib.load(os.path.join(weight_root, 'svmclf.pkl'))
  ####

  #### start visual webcam and SenXor thermal cam
  cam = cv.VideoCapture(webcam_id)
  params = {'regwrite': [(0xB4, 0x03), (0xD0, 0x00),(0x30, 0x00), (0x25, 0x00)],
            'sens_factor': 95,'offset_corr': 1.5,'emissivity': 97}

  mi48 = config_mi48(params)
  ncols, nrows = mi48.fpa_shape
  mi48.start(stream=True, with_header=True)
  ####

  #### STARK PARAMETERS HERE:
  minav = RollingAverageFilter(N=15)
  maxav = RollingAverageFilter(N=8)
  minav2 = RollingAverageFilter(N=15)
  maxav2 = RollingAverageFilter(N=8)

  frame_filter = STARKFilter({'sigmoid': 'sigmoid','lm_atype': 'ra','lm_ks': (3,3),
                              'lm_ad': 6,'alpha': 2.0,'beta': 2.0,})
  ####

  #### HOMOGRAPHY MATRIX HERE:
  with open('homography_alignment/homographymatrix.json', "r") as f:
    local_M_120120 = json.load(f)
    local_M_120120 = np.array(local_M_120120['matrix']).astype(np.float64)
  
  RSCALE = 2
  ####

  cv.namedWindow("Display")

  empty_frame = np.zeros((120,120,3))

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
        
        yy = thermalcrop[y_coords[4],:]   # thermal horizontal cut from nose landmark point

        if np.std(yy/np.max(yy)) > liveness_threshold and \
        np.std(thermalcrop.flatten().astype(np.float32)) > heat_threshold:
          pred = FC2_predict(thermalcrop, rgbcrop, (x_coords, y_coords), model, SVMclf)
          color = (0, 255, 0) if pred else (0, 0, 255)
          rgbimgbig = cv.rectangle(rgbimgbig, (xminbig, yminbig), (xmaxbig, ymaxbig), color, 1)
          # print(f"First threshold passed. Second threshold{'' if pred else ' NOT'} passed.")
        else:
          rgbimgbig = cv.rectangle(rgbimgbig, (xminbig, yminbig), (xmaxbig, ymaxbig), (0, 0, 255), 1)
          # print("NO threshold passed")

      else:
        rgbimgbig = cv.resize(rgbimg, dsize=None, fx=RSCALE, fy=RSCALE)
        thermalimgbig = cv.resize(thermalimg, dsize=None, fx=RSCALE, fy=RSCALE)

    cv.imshow("Display", np.vstack((rgbimgbig, thermalimgbig)))
      # end = time.time()
      # print(f"FPS: {total_frames/(end-start)}")   # tracking FPS deterioration/stability

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
          # total_frames = 0
          # start = time.time()
          break

  cv.destroyAllWindows()
