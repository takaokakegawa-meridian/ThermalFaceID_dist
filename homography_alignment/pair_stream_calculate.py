"""
File: homography_alignment/pair_stream_calculate.py
Author: Takao Kakegawa
Date: 2024
Description: Script is to enable streaming of the created/saved pair images.
             You will need a separate excel sheet open to input the values as you stream images.
             The empty excel spreadsheet template is included in the repository
"""

import os
import traceback
import argparse
from homography import view_display


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for homography coordinates')
    parser.add_argument('-num', type=int, default=4, help='number of image pairs to evaluate')
    args = parser.parse_args()
    num_pairs = args.num

    save_root = os.getcwd()
    thermal_root = os.path.join(save_root, 'homography_sampleset/thermal')
    webcam_root = os.path.join(save_root, 'homography_sampleset/visual')

    try:
        for filename in os.listdir(thermal_root)[:num_pairs]:
            print(f"Now viewing file: {filename}")
            view_display(thermal_root, webcam_root, filename)
            print(f"Finished image file {filename}, Next one ...")

        print(f"Finished viewing {num_pairs} images.")
        
    except Exception as e:
        print("Error processing. Please ensure previous steps done correctly. Error traceback:")
        print(traceback.format_exc())
        exit()

    exit()
    