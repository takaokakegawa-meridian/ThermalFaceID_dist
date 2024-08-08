"""
File: val_procedure/val_eval.py
Author: Takao Kakegawa
Date: 2024
Description: Main script to run to observe plain CV window demonstrating Facial Anti-Spoofing with thermal data.
"""

import ast
import argparse
import joblib
import os
import sys
sys.path.append(os.getcwd())
import json
import tomllib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2 as cv
import torch
from Depth_FCN_2.FCN import DepthBasedFCN      # FCN2 model imports

# modularised imports
from thermalfaceid.processing import *
from thermalfaceid.utils import *
from thermalfaceid.inference import *
from utils import *


rotation_map = {'90': cv.ROTATE_90_CLOCKWISE,
                '-90': cv.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv.ROTATE_180}

with open("config.toml", "rb") as f:
  config = tomllib.load(f)

if __name__ == "__main__":
    #### Load in config default params
    default_params = config['tool']['model_params']
    ####

    #### Load both Patch-FCN model and SVM classifiers
    weight_root = "Depth_FCN_2/model_res"
    model = DepthBasedFCN(3)
    model.load_state_dict(torch.load(os.path.join(weight_root,"best_weights.pt"),
                        map_location=torch.device('cpu')))
    print("[CHECK] Fully-Convoluted Network loaded in successfully ...")
    model.eval()
    SVMclf = joblib.load(os.path.join(weight_root, 'svmclf.pkl'))
    print("[CHECK] SVM Classifier loaded in successfully ...")
    ####

    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    target_dir = os.path.join(desktop_path, 'thermalfaceid_val')

    if check_structure(target_dir):
       print("Validation dataset folder structure validated ...")
    else:
       sys.exit("incorrect structure. Please check. Exiting script.")

    real_dir = os.path.join(target_dir, "real")
    fake_dir = os.path.join(target_dir, "fake")

    real_filenames = get_common_filenames(real_dir)
    print(f"Number of 'real' labeled datapoints: {len(real_filenames)}")
    fake_filenames = get_common_filenames(fake_dir)
    print(f"Number of 'fake' labeled datapoints: {len(fake_filenames)}")

    print(real_filenames[:3])

    if not (check_common_files(real_filenames, os.path.join(real_dir, "thermal"),
                          os.path.join(real_dir, "rgb"), os.path.join(real_dir, "landmarkcoords")) and \
            check_common_files(fake_filenames, os.path.join(fake_dir, "thermal"),
                          os.path.join(fake_dir, "rgb"), os.path.join(fake_dir, "landmarkcoords"))):
       print("common file-presence check failed. Exiting script.")
       sys.exit()
    else:
       print("common file-presence check passed ...")

    