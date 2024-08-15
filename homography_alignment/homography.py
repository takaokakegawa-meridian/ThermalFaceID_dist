"""
File: homography_alignment/homography.py
Author: Takao Kakegawa
Date: 2024
Description: Script with all homography related functions
"""

import os
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def homographic_blend(img_src: np.ndarray, img_dst: np.ndarray, M: np.ndarray) -> np.ndarray:
    """project a source image onto a destination plane homography.
    Args:
        img_src (np.ndarray): 3-channel source image that we will be transforming/changing.
        img_dst (np.ndarray): 3-channel destination image whose plane we want to project onto.
        M (np.ndarray): 2D (3x3) homography matrix to translate img_src onto img_dst plane.
    Returns:
        np.ndarray: 3-channel image of the source image homographically transformed.
    """
    rows,cols, _ = img_dst.shape  
    dst = cv.warpPerspective(img_src, M, (cols, rows))
    return dst


def homographic_blend_alpha(img_src: np.ndarray, img_dst: np.ndarray,
                            M: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """overlay a source image onto a destination image's plane using homography with opacity alpha.
    Args:
        img_src (np.ndarray): 3-channel source image that we will be transforming/changing.
        img_dst (np.ndarray): 3-channel destination image whose plane we want to overlay onto.
        M (np.ndarray): 2D (3x3) homography matrix to translate img_src onto img_dst plane.
        alpha (float, optional): opacity of img_src on img_dst. Defaults to 0.3.
    Returns:
        np.ndarray: 3-channel image of img_src image homographically transformed to img_dst.
    """
    rows,cols, _ = img_dst.shape  
    dst = cv.warpPerspective(img_src, M, (cols, rows))
    overlay = cv.addWeighted(img_dst, alpha, dst, 1-alpha, 0)
    return overlay


def homographic_blend_alpha_inv(img_src: np.ndarray, img_dst: np.ndarray,
                                M: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """same functionality as homographic_blend_alpha, but in the inverse. The matrix M is still
    representative of the transformation for img_src to img_dst.
    Args:
        img_src (np.ndarray): 3-channel source image whose plane we want to overlay onto.
        img_dst (np.ndarray): 3-channel destination image that we will be transforming/changing.
        M (np.ndarray): 2D (3x3) homography matrix to translate img_src onto img_dst plane.
        alpha (float, optional): opacity of img_src on img_dst. Defaults to 0.3.
    Returns:
        np.ndarray: 3-channel image of img_dst homographically transformed to img_src.
    """
    rows,cols, _ = img_src.shape  
    dst = cv.warpPerspective(img_dst, M, (cols, rows), flags=cv.WARP_INVERSE_MAP)
    overlay = cv.addWeighted(img_src, alpha, dst, 1-alpha, 0)
    return overlay


def homographic_encode(thermal_img: np.ndarray, webcam_img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """add thermal_frame as additional channel via homographic overlay.
    Args:
        thermal_img (np.ndarray): 2D array of the thermal frame. Should be grayscaled.
        webcam_img (np.ndarray): 3D array of the visual frame in BGR format
        M (np.ndarray): 2D (3x3) homography matrix
    Returns:
        np.ndarray: 4D array of the webcam image with thermal in BGRT format.
    """
    rows,cols, _ = webcam_img.shape  
    dst = cv.warpPerspective(thermal_img, M, (cols, rows))
    print(f"dstack: {thermal_img}")
    webcam_img[:,:,-1] = dst[:,:,0]
    return webcam_img


def view_display(thermal_root: str, webcam_root: str, filename: str) -> None:
    """function to display the thermal/visual image pair in a single matplotlib window.
    Args:
        thermal_root (str): thermal images directory path
        webcam_root (str): webcam images directory path
        filename (str): the filename of the webcam/thermal images
    """
    _, ax = plt.subplots(figsize=(12,6),ncols=2)
    img1 = cv.imread(os.path.join(thermal_root,filename))
    img2 = cv.imread(os.path.join(webcam_root,filename))
    img1, img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB), cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[0].set_title(filename)
    ax[1].set_title(filename)
    plt.show()


def euclid_distance(x1: float, y1: float, x2: float, y2: float) -> float:
  """calculate euclidean distance between two points (x1, y1) and (x2, y2)
  Args:
    x1 (float): x-coord of point p1
    y1 (float): y-coord of point p1
    x2 (float): x-coord of point p2
    y2 (float): y-coord of point p2
  Returns:
    float: euclidean distance between two points
  """
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def closest_point(target, idxs_kp_lst):
  if len(idxs_kp_lst) < 1:
    return None
  
  heap = []

  for idx_kp in idxs_kp_lst:
    dist = euclid_distance(target[0], target[1], idx_kp[1].pt[0], idx_kp[1].pt[1])
    heapq.heappush(heap, (dist, idx_kp))

  return heapq.heappop(heap)[1]


def get_contour_keypoints_descriptors(kps, deses, contours, centroids):
    assert len(kps) == len(deses), "keypoint and descriptor lists must be same length."
    kp_d = defaultdict(list)

    for kp_idx, kp in enumerate(kps):    # keep track of kp_idx to get corresponding descriptor
        for c_idx, c in enumerate(contours):    # keep track of c_idx to get corresponding centroid
            if cv.pointPolygonTest(c, (int(kp.pt[0]), int(kp.pt[1])), False) == 1:
                kp_d[c_idx].append((kp_idx, kp))

    kp_final = []

    for c_idx, c in enumerate(contours):
        c_res = closest_point(centroids[c_idx], kp_d[c_idx])
        kp_final.append(c_res)
    
    return [i[1] for i in kp_final if i is not None], deses[[i[0] for i in kp_final if i is not None]]


def binarize_img(img: np.ndarray, revert=False, blockSize=5, C=6.) -> np.ndarray:
    """function to binarize image using adaptive-mean thresholding
    Args:
        img (np.ndarray): input 3-channel img
        revert (bool, optional): True is you want to revert grayscale. Defaults to False.
        blockSize (int, optional): blockSize parameter for cv.AdaptiveThreshold. Defaults to 5.
        C (float, optional): blockSize parameter for cv.AdaptiveThreshold. Defaults to 6.
    Returns:
        np.ndarray: _description_
    """
    res = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    adaptive_threshold_mean = cv.adaptiveThreshold(res, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                                   cv.THRESH_BINARY, blockSize, C)

    if revert:
        return 255 - adaptive_threshold_mean

    return adaptive_threshold_mean


def get_flann_matches(flann, des1, des2, MIN_MATCH_COUNT, k=2):
    matches = flann.knnMatch(des1, des2, k)
    good = []
    
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return good if len(good) >= MIN_MATCH_COUNT else None
