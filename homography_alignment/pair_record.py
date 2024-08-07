"""
File: homography_alignment/pair_record.py
Author: Takao Kakegawa
Date: 2024
Description: Script collect RGB/Thermal image pairs as part of set up process to calibrate the 
             homography matrix. his script must be executed in the root directory, i.e. at 
             ThermalFaceID_dist level.
"""

import sys
import os
sys.path.append(os.getcwd())        # for relative imports in root directory

import numpy as np
import cv2 as cv
import argparse

# SenXor imports
from senxor.utils import get_default_outfile, remap
from senxor.filters import RollingAverageFilter
from senxor.display import cv_render

# modularised imports
from thermalfaceid.stark import STARKFilter
from thermalfaceid.processing import process_thermal_frame
from thermalfaceid.utils import config_mi48


rotation_map = {'90': cv.ROTATE_90_CLOCKWISE,
                '-90': cv.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv.ROTATE_180}


if __name__ == "__main__":
    #### Create the argument parser
    parser = argparse.ArgumentParser(description='Argument Parser for Webcam ID and Rotation')
    parser.add_argument('-webcam_id', type=int, default=0, help='Webcam ID if default detected webcam is not Logitech cam')
    parser.add_argument('-rotation', type=int, default=90, help='Rotation for webcam if needed')
    parser.add_argument('-colormap', type=str, default="inferno", help='Thermal image colormap')
    args = parser.parse_args()
    webcam_id = args.webcam_id
    rotation = rotation_map.get(str(args.rotation), None)
    cmap = args.colormap

    #### create/find save root for both thermal/visual images.
    saveroot = os.path.join(os.getcwd(),'homography_sampleset')
    if not os.path.isdir(saveroot):
        os.mkdir(saveroot)
        os.mkdir(os.path.join(saveroot,'thermal'))
        os.mkdir(os.path.join(saveroot,'visual'))
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

    data, header = mi48.read()
    if (data is None) or not (cam.isOpened()):
        mi48.stop()
        cv.destroyAllWindows()
        sys.exit(1)

    else:
        empty_frame = np.zeros((113,102,3))
        empty_frame2 = np.zeros((480,480,3))
        save_imgs = False
        cv.namedWindow("Visual Image")
        cv.namedWindow("Thermal Image")

        while True:
            ret, rgbframe = cam.read()
            data, _ = mi48.read()
            if ret and data is not None:
                rgbimg = np.fliplr(cv.rotate(rgbframe, rotation))[125:605,:]
                thermal_raw = process_thermal_frame(data, ncols, nrows, minav, maxav,
                                                    minav2, maxav2, frame_filter)
                # thermal_frame = thermal_raw[:113,26:128]    # thermal and projected rgb have same dimensions (113, 102)
                thermal_frame = thermal_raw[:,20:-20]
                thermalimg = cv_render(remap(thermal_frame),
                                       resize=(int(thermal_frame.shape[1]),int(thermal_frame.shape[0])),
                                       colormap=cmap,
                                       interpolation=cv.INTER_NEAREST_EXACT,
                                       display=False)

                cv.imshow("Visual Image",rgbimg)
                cv.imshow("Thermal Image",cv.resize(thermalimg, dsize=rgbimg.shape[:2],
                                                    interpolation=cv.INTER_CUBIC))
                
                if save_imgs == True:
                    savename = get_default_outfile()
                    cv.imwrite(os.path.join(os.path.join(saveroot,'thermal'), savename),
                            thermalimg)
                    cv.imwrite(os.path.join(os.path.join(saveroot,'visual'), savename),
                            rgbimg)

            else:
                cv.imshow("Visual Image",empty_frame)
                cv.imshow("Thermal Image",empty_frame2)

            key = cv.waitKey(1)  # & 0xFF
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
                save_imgs = True
                print("Save images ENABLED")
            elif key == ord("n"):
                save_imgs = False
                print("Save images DISABLED")

    # stop capture and quit
    mi48.stop()
    cv.destroyAllWindows()
