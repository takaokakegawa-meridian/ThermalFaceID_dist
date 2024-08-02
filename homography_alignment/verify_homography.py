"""
File: verify_homography.py
Author: Takao Kakegawa
Date: 2024
Description: Script to verify the calculated homography by displaying the overlay in a CV window.
             The homography matrix will be calculated, then saved into a json file 
"""

import os
import traceback
import csv
import argparse
import json
import cv2 as cv
import numpy as np

from homography import homographic_blend_alpha

save_root = os.path.join(os.getcwd(), 'homography_alignment')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for homography verification')
    parser.add_argument('-filename', type=str, default="homography_coord.csv",
                        help='name of homography csv file, e.g. homography_coord.csv')
    args = parser.parse_args()
    filename = args.filename
    filepath = os.path.join(save_root, filename)

    all_Ms = np.zeros((3,3))

    try:
        print("Calculating homography matrix ...")
        with open(filepath, 'r') as file:
            my_reader = csv.reader(file, delimiter=',')
            for idx, row in enumerate(my_reader):
                if idx != 0:
                    row = np.array(row).reshape(8,2).astype(np.float16)
                    thermalrow, webcamrow = row[:4], row[4:]
                    h, _ = cv.findHomography(webcamrow, thermalrow)
                    all_Ms += h

        all_Ms /= idx

        print("Completed calculation of homography matrix ...")
        print(all_Ms)
        print("---"*20)

        with open(os.path.join(save_root, 'homographymatrix.json'), 'w') as f:
            json.dump({'matrix': all_Ms.tolist()}, f)

        thermal_root = os.path.join(os.getcwd(), 'homography_sampleset/thermal')
        webcam_root = os.path.join(os.getcwd(), 'homography_sampleset/visual')
        samplefile = os.listdir(thermal_root)[0]
        thermalimg = cv.imread(os.path.join(thermal_root,samplefile))
        webcamimg = cv.imread(os.path.join(webcam_root,samplefile))
        displayimg = cv.resize(homographic_blend_alpha(webcamimg, thermalimg, all_Ms, 0.6),
                            dsize=None, fx=2, fy=2, interpolation=cv.INTER_NEAREST_EXACT)

        print("Observe sample overlay with calculated homography matrix ...")
        print("press 'q' if you want to exit.")

        while True:
            cv.imshow("Display", displayimg)
            key = cv.waitKey(1)  # & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            
        cv.destroyAllWindows()
        exit()

    except Exception as e:
        print("Error processing. Please ensure previous steps done correctly. Error traceback:")
        print(traceback.format_exc())
        exit()
