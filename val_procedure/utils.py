
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
from typing import List

import numpy as np
import cv2 as cv
from senxor.utils import get_default_outfile
from thermalfaceid.processing import convert_bgr_cnn_input
import torch
from torch.utils.data import Dataset


def check_structure(folder: str) -> bool:
  """function to check inner-level structure of folder path is correct for validation data collection.
  Args:
      folder (str): parent-level folder path
  Returns:
      bool: True if structure is correct, False otherwise.
  """
  # Checks that the directory's structure matches validation collection needs.
  expected_structure = ['real', 'fake']
  for subdir in expected_structure:
    if not os.path.exists(os.path.join(folder, subdir)):
      return False
    for subsubdir in ['thermal', 'rgb', 'landmarkcoords']:
      if not os.path.exists(os.path.join(folder, subdir, subsubdir)):
        return False
  return True


def create_folders(base_dir: str) -> None:
  """Function to create local directory of right structure for user to collect data.
    Args:
        base_dir (str): base directory path as string
  """
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


def get_common_filenames(dir_path: str) -> List[str]:
  """Get the intersection of filenames in three folders within a directory.
    Args:
      dir_path (str): The path to the directory containing "rgb", "thermal", and "landmarkcoords" folders.
    Returns:
      list: A list of filenames that are common to all three folders.
  """
  rgb_files = set([i[:-3] for i in os.listdir(os.path.join(dir_path, "rgb")) if i.endswith(".png")])
  thermal_files = set([i[:-4] for i in os.listdir(os.path.join(dir_path, "thermal")) if i.endswith(".json")])
  landmark_files = set([i[:-4] for i in os.listdir(os.path.join(dir_path, "landmarkcoords")) if i.endswith(".json")])

  common_files = rgb_files.intersection(thermal_files, landmark_files)
  
  return list(common_files)


def check_common_files(filenames: List[str], thermal_root: str, rgb_root: str,
                       landmark_coords_root: str) -> bool:
  """function to check that the list of filenames is commonly found in all three roots.
  Args:
      filenames (List[str]): list of filenames
      thermal_root (str): path to JSON files containing thermalcrop information 
      rgb_root (str): path to .PNG files containing RGB input.
      landmark_coords_root (str): _description_
  Returns:
      bool: _description_
  """
  if len(filenames) == 0:
    return True

  try:
    thermalfileinter = set([i + "json" for i in filenames]).issubset(set([i for i in os.listdir(thermal_root) if i.endswith(".json")]))
    rgbfileinter = set([i + "png" for i in filenames]).issubset(set([i for i in os.listdir(rgb_root) if i.endswith(".png")]))
    landmarkfileinter = set([i + "json" for i in filenames]).issubset(set([i for i in os.listdir(landmark_coords_root) if i.endswith(".json")]))
    assert thermalfileinter and rgbfileinter and landmarkfileinter, "Some files not found in rgb, thermal, landmarkcoords root dirs."
    return True

  except Exception as e:
    print("One or more paths not found")
    return False

# class val_Dataset(Dataset):
#   """Custom torch.Dataset class to acccomodate validation data collected for FCN model validation evaluation.
#   """

#   def __init__(self, thermal_root: str, rgb_root: str, landmark_coords_root: str,
#     filenames: List[str]):

#     try:
#       # directory for HSV+YCrCb input images.
#       self.rgb_root = rgb_root
#       self.thermal_root = thermal_root
#       self.landmark_coords_root = landmark_coords_root
#       self.filenames = filenames

#       thermalfileinter = set([i+"json" for i in self.filenames]).issubset(set([i for i in os.listdir(self.thermal_root) if i.endswith(".json")]))
#       rgbfileinter = set([i+"png" for i in self.filenames]).issubset(set([i for i in os.listdir(self.rgb_root) if i.endswith(".png")]))
#       landmarkfileinter = set([i+"json" for i in self.filenames]).issubset(set([i for i in os.listdir(self.landmark_coords_root) if i.endswith(".json")]))
#       assert(thermalfileinter and rgbfileinter and landmarkfileinter), "some files not found in rgb, thermal, landmarkcoords root dirs."

#     except Exception as e:
#       print("one or more paths not found")


#   def __len__(self):
#     return len(self.filenames)


#   def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.Tensor] | None:
#     """__getitem__ method to get desired inputs for model training and evaluation.
#     Args:
#     idx (int): index of data to get.
#     Returns:
#     tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.Tensor]: returns actual thermal landmark 
#     values, 3-channel visual image, the visual image dimensions, and the landmarks coordinates for the
#     visual image.
#     """
#     try:
#       fn = self.filenames[idx]
#     except IndexError:
#       print(f"{idx} is an invalid index")
#       return

#     with open(os.path.join(self.thermal_root, fn+"json"), "r") as f:
#       thermal_vals = np.array(json.load(f)['thermalcrop'])

#     with open(os.path.join(self.landmark_coords_root, fn+"json"), "r") as f:
#       dat = json.load(f)
#       x_coords, y_coords = np.array(dat['x_coords']).astype(int), np.array(dat['y_coords']).astype(int)

#     rgb_img = cv.imread(os.path.join(self.rgb_root, self.file_names[idx]))
#     rgb_shape = rgb_img.shape[:2]
#     thermal_vals = torch.from_numpy(thermal_vals).float()
#     rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
#     landmarks = np.array(self.landmark_coords[self.file_names[idx]]["rgb"])
#     landmarks = torch.from_numpy(landmarks).int()
#     assert landmarks.shape[0] == thermal_vals.shape[0], "mismatch with number of landmarks"
#     return thermal_vals, rgb_img, rgb_shape, landmarks