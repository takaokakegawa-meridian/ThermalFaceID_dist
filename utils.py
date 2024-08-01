from senxor.utils import connect_senxor
from senxor.mi48 import MI48
import numpy as np


def config_mi48(params: dict) -> MI48:
  """configure mi48 object to register parameters passed in as dictionary
  Args:
    params (dict): mi48 reigster parameters to write in, as a dictionary.
  Returns:
    MI48: detected mi48 object that has been connected, registered parameters written in.
  """
  mi48 = connect_senxor()
  for reg in params['regwrite']:
    mi48.regwrite(*reg)
  if 'sens_factor' in params.keys():
    mi48.set_sens_factor(params['sens_factor'])
  if 'offset_corr' in params.keys():
    mi48.set_sens_factor(params['offset_corr'])
  if 'emissivity' in params.keys():
    mi48.set_sens_factor(params['emissivity'])
  return mi48

def euclid_distance(x1: float, y1: float, x2: float, y2: float) -> float:
  """calculate euclidean distance between two points (x1, y1) and (x2, y2)
  Args:
    x1 (int): x-coord of point p1
    y1 (int): y-coord of point p1
    x2 (int): x-coord of point p2
    y2 (int): y-coord of point p2
  Returns:
    float: euclidean distance between two points
  """
  return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def calculate_gamma(x_coords: np.ndarray, y_coords: np.ndarray, thermalcrop: np.ndarray) -> float:
  """calculate 'wavelength beat' for given frame.
  Args:
    x_coords (np.ndarray): mediapipe facial landmark x-coordinates corresponding to thermalcrop dimensions
    y_coords (np.ndarray): mediapipe facial landmark y-coordinates
    thermalcrop (np.ndarray): cropped thermal frame of face
  Returns:
    float: wavelength beat value
  """
  d = euclid_distance(x_coords[468],y_coords[468],x_coords[473],y_coords[473])/4
  eyelinecenter = (np.mean(x_coords[[468,473]]),np.mean(y_coords[[468,473]]))
  roi_starty = int(eyelinecenter[1]-d)
  roi_endy = int(eyelinecenter[1]-5*d/2)
  roi_startx = int(eyelinecenter[0]-3*d/2)
  roi_endx = int(eyelinecenter[0]+3*d/2)
  beat = np.mean(thermalcrop[roi_endy:roi_starty+1,roi_startx:roi_endx+1])
  return beat