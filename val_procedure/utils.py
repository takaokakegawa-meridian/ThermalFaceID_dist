
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
      for subsubdir in ['thermal', 'rgb']:
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
  os.makedirs(os.path.join(base_dir, 'fake', 'thermal'))
  os.makedirs(os.path.join(base_dir, 'fake', 'rgb'))
  print("Directory with right structure created ...")
  return
