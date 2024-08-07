
"""
File: val_procedure/utils.py
Author: Takao Kakegawa
Date: 2024
Description: Script utility functions for validation dataset collection procedure to
             ultimately generate some statistics on model performance.
"""

import os
import sys
import shutil
import json

import numpy as np
import cv2 as cv
from senxor.utils import get_default_outfile


def create_folders(base_dir: str) -> None:
  """Function to create local directory of right structure for user to collect data.
    Args:
        base_dir (str): base directory path as string
  """
  def check_structure(folder):
    # Checks that the directory's structure matches validation collection needs.
    expected_structure = ['real', 'fake']
    for subdir in expected_structure:
      if not os.path.exists(os.path.join(folder, subdir)):
        return False
      for subsubdir in ['thermal', 'rgb', 'landmarkcoords']:
        if not os.path.exists(os.path.join(folder, subdir, subsubdir)):
          return False
    return True

  if os.path.exists(base_dir):
    if not check_structure(base_dir):
      response = input("Folder name already exists but different structure. Confirm replace? (Y/N): ")
      if response.lower() == 'y':
        print("Replacing directory into right structure ...")
        shutil.rmtree(base_dir)
      else:
        sys.exit("Exiting script. Folder not recreated.")
    else:
      print("Directory already exists with right structure.")
      return
  os.makedirs(base_dir)
  os.makedirs(os.path.join(base_dir, 'real', 'thermal'))
  os.makedirs(os.path.join(base_dir, 'real', 'rgb'))
  os.makedirs(os.path.join(base_dir, 'real', 'landmarkcoords'))
  os.makedirs(os.path.join(base_dir, 'fake', 'thermal'))
  os.makedirs(os.path.join(base_dir, 'fake', 'rgb'))
  os.makedirs(os.path.join(base_dir, 'fake', 'landmarkcoords'))
  print("Directory with right structure created ...")
  return


def save_val_data(save_dir: str, x_coords: np.ndarray, y_coords: np.ndarray,
                  rgbcrop: np.ndarray, thermalcrop: np.ndarray) -> None:
  """function to save the recorded validation data into validation directory

  Args:
      save_dir (str): higher-level save directory
      x_coords (np.ndarray): array of landmark x-coordinates
      y_coords (np.ndarray): array of landmark y-coordinates
      rgbcrop (np.ndarray): cropped visual image as 3-channel RGB array
      thermalcrop (np.ndarray): cropped thermal frame as 2D array
  """
  savename = get_default_outfile(ext="")
  thsavedir = os.path.join(save_dir, "thermal")
  rgbsavedir = os.path.join(save_dir, "rgb")
  landmarksavedir = os.path.join(save_dir, "landmarkcoords")

  with open(os.path.join(landmarksavedir, savename+"json"), "w") as f:
    json.dump({"x_coords": x_coords.tolist(), "y_coords": y_coords.tolist()}, f)

  with open(os.path.join(thsavedir, savename+"json"), "w") as f:
    json.dump({"thermalcrop": thermalcrop.tolist()}, f)

  cv.imwrite(os.path.join(rgbsavedir, savename+"png"), rgbcrop)
  return
