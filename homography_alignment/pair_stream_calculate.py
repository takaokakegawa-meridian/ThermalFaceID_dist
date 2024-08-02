#### this script is to enable streaming of the created/saved pair images.
#### you will need a separate excel sheet open to input the values as you stream images
#### the empty excel spreadsheet template is included in the repository

import os
import traceback
import argparse
import cv2 as cv
import matplotlib.pyplot as plt

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
    