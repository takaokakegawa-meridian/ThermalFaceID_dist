"""
File: homography_alignment/homography.py
Author: Takao Kakegawa
Date: 2024
Description: Script with all homography related functions
"""

import os
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