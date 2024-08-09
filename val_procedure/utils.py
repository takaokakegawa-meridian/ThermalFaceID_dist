
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
from sklearn.pipeline import Pipeline
from Depth_FCN_2.FCN import DepthBasedFCN
from thermalfaceid.inference import FC2_predict
from senxor.utils import get_default_outfile


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


def get_results(filenames: List[str], thermal_root: str, rgb_root: str, landmark_coords_root: str,
                actual_lab: int, NN_model: DepthBasedFCN, SVMclf: Pipeline):   # need to specify output types later
  
  if len(filenames) == 0:
    return {}
  labelmapping = {(1,1): "true pos", (0,0): "true neg",(1,0): "false pos", (0,1): "false neg"}
  d = {}

  if actual_lab == 1:
    tp, fn = 0, 0
  elif actual_lab == 0:
    tn, fp = 0, 0

  for filename in filenames:
    with open(os.path.join(thermal_root, filename+"json"), "r") as f:
      thermalcrop = np.array(json.load(f)['thermalcrop'])

    with open(os.path.join(landmark_coords_root, filename+"json"), "r") as f:
      dat = json.load(f)
      x_coords, y_coords = np.array(dat['x_coords']).astype(int), np.array(dat['y_coords']).astype(int)
      landmarkcoords = (x_coords, y_coords)

    rgbimg = cv.imread(os.path.join(rgb_root, filename+"png"))

    res = 1 if FC2_predict(thermalcrop, rgbimg, landmarkcoords, NN_model, SVMclf) else 0
    labelmap = labelmapping[(res, actual_lab)]
    d[filename] = {"actual": actual_lab, "predicted": res, "label": labelmap}

    if labelmap == "true pos":
      tp += 1
    elif labelmap == "true neg":
      tn += 1
    elif labelmap == "false pos":
      fp += 1
    elif labelmap == "false neg":
      fn += 1
  
  summ = {"true pos": tp, "false neg": fn} if actual_lab == 1 else {"true neg": tn, "false pos": fp}
  return d, summ


def get_summary_statistics(d: dict):
  total = sum(d.values())
  acc = (d["true pos"] + d["true neg"])/total
  precision = d["true pos"]/(d["true pos"] + d["false pos"])
  recall = d["true pos"]/(d["true pos"] + d["false neg"])
  fpr = d["false pos"]/(d["false pos"] + d["true neg"])
  fnr = d["false neg"]/(d["false neg"] + d["true pos"])
  f1_score = 2 * precision * recall / (precision + recall)
  print(f"accuracy: {acc}, precision: {precision}, recall: {recall}, f1-score: {f1_score}")
  print(f"false-pos rate: {fpr}, false-neg rate: {fnr}")
  summ = {"accuracy": acc, "precision": precision, "recall": recall, "false-pos rate": fpr,
          "false-neg rate": fnr, "f1-score": f1_score}
  return summ
