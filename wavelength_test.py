# https://www.techscience.com/cmc/v68n3/42515/html
#### script to test the potential of tracking waveform pulses via thermal imaging.

import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2 as cv
from homography_alignment.homography import homographic_blend

# Mediaipipe facial landmark imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SenXor imports
from senxor.filters import RollingAverageFilter
from stark import STARKFilter

# modularised imports
from processing import process_thermal_frame
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
  parser.add_argument('-confidence', type=int, default=0.7, help='facial landmark detection confidence threshold')
  parser.add_argument('-height_ratio', type=float, default=0.77, help='minimum height ratio of frame for face to occupy')
  args = parser.parse_args()
  webcam_id = args.webcam_id
  min_height_ratio = args.height_ratio
  rotation = rotation_map.get(str(args.rotation), None)
  ####

  #### Create FaceDetection and FaceLandmarker objects.
  landmarkoptions = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='mediapipe_models/face_landmarker_v2_with_blendshapes.task'),
                                                 output_face_blendshapes=True,
                                                 output_facial_transformation_matrixes=True,
                                                 num_faces=1)
  landmarker = vision.FaceLandmarker.create_from_options(landmarkoptions)
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
  local_M_120120 = np.array([[ 2.88807358e-01,  4.71680005e-02, -1.23089966e+01],
                             [-6.03625342e-03,  3.35243264e-01, -1.77408347e+00],
                             [-4.91781884e-06,  7.87303985e-04,  1.00000000e+00]])
  
  RSCALE = 2
  ####
  fig, ax = plt.subplots()

  cv.namedWindow("Display")
  cv.namedWindow("Display2")

  empty_frame = np.zeros((120,120,3))

  xx = []

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

        # cv.imshow("Display2", plot_img)
        d = euclid_distance(x_coords[468],y_coords[468],x_coords[473],y_coords[473])/4
        # print(f"distance (d): {d}")
        eyelinecenter = (np.mean(x_coords[[468,473]]),np.mean(y_coords[[468,473]]))
        roi_starty = int(eyelinecenter[1]-d)
        roi_endy = int(eyelinecenter[1]-5*d/2)
        roi_startx = int(eyelinecenter[0]-3*d/2)
        roi_endx = int(eyelinecenter[0]+3*d/2)

        for i in range(478):
          thermalimgbig = cv.circle(thermalimgbig, (x_coordsbig[i],y_coordsbig[i]), 1, (0,0,0), -1)
        # cv.rectangle(rgbimgbig, (roi_startx, roi_starty), (roi_endx, roi_endy), (255,0,0), 1)

        ax.clear()
        ax.set_ylim(25,37)
        beat = np.mean(thermalcrop[roi_endy:roi_starty+1,roi_startx:roi_endx+1])
        if len(xx) == 4:
          xx.pop(0)
          xx.append(beat)
          ax.plot(range(len(xx)),xx)
          ax.set_title(f"std: {np.std(xx)}")
          fig.canvas.draw()
          plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
          plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
          cv.imshow("Display2", plot_img)
        else:
          xx.append(beat)

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
