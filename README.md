# Thermal-Visual Dual modal anti-spoofing/verification development:

Meridian Innovation development project for potential anti-spoofing/facial identity verification application with our highest resolution thermal sensor, **Panther** ($120\times160$).

## Contents:
- [Introduction](#introduction)
- [Set Up Guide](#set-up-guide)
    - [Hardware](#hardware)
    - [Software Environment](#software-environment)
    - [Homography calibration](#homography-calibration)
- [Running Inference](#running-inference)
- [Miscellaneous Analytics](#miscellaneous-analytics)
- [Analysis and Results](#analysis-and-results)

## Introduction:
This directory includes the work of the second iteration of the Depth-Based method inspired by [Liu et al., 2017](https://cvlab.cse.msu.edu/pdfs/FaceAntiSpoofingUsingPatchandDepthBasedCNNs.pdf) that utilises fully convoluted networks to generate estimated thermal mappings, starting with a visual RGB image.

Building upon the first iteration, the second version of our facial anti-spoofing has diverged from using the YOLO framework, and instead utilises [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Gemini AI), which offers facial landmark detection with a model that is both memory/computationally more favourable.

## Set Up Guide:
### Hardware:
In order to use this, there are a few things to set up. Firstly, you will need the following hardware:
- Logitech C270 HD Webcam
- Panther 45
- SenXor MI48 compatbile with Panther
- Camera stand/mount
- 3D printed board for this set up (provided file)
- $6\times4$ grid-patterned heat tile with stand to mount on.

For this set up, we have generated a custom board to mount both the visual/thermal cameras with their boards. This board can be 3D-printed using the file: `3dmountprint.stl`. Thank you to Quincy Choy for collaborating with continuous feedback to design this mount. The board is to be mounted on the camera stand.

<p align="center">
    <img src="pictures/hardwaresetup.JPG" alt="Hardware Setup" width="30%">
</p>
<p align="center">
    <i>Ideal set up with the 3D printed mount holding both Panther and Logitech C270 HD webcam.</i>
</p>

### Software Environment:
**This will be ideal to set up with Python 3.11**. Any version of Python 3.10+ should suffice. In a virtual environment, install all the necessary packages and libraries by running `pip install -r requirements.txt` from the root directory. You may be prompted to log in to GitHub to install pysenxor. 

### Homography calibration:
Homography alignment can now be done in a more streamlined, automated manner. We have constructed a set up procedure that requires less steps/effort than the previous approach outlined, and is more reliable in calibration as it utilises particular landmark points that are clearer to detect. For this step, all necessary scripts are to be found in the `/homography_alignment` folder.

We are using homography to align the two different image planes we have. Homography provides a projective transformation with a $3\times3$ matrix mapping two planar projections, to describe the relative motions observed in one plane, onto another. Now ideally, simple homography is meant to be done with the same cameras, as there are some intrinstic camera parameters that impact the homography effectiveness. For the meantime, we are assuming that the properties are the same/similar enough such that there is not a significant difference. The euclidean homography matrix is converted to a projective matrix ($a\rightarrow b$) through the expression $H'_{a,b}=KH_{a,b}K^{-1}$, where $K$ is the common intrinsic matrix shared by both cameras $a, b$.

However, this needs to be further reviewed later. If it is the case that this can be improved, there are already considerations for how to mitigate the issues, such as grid-based patch homography, or pre-calibration to determine the intrinsic properties of each camera. Pre-calibration can give us  estimates for $\widehat{K_a}, \widehat{K_b}$ to which we get our homography estimate $\widehat{H'_{a,b}}=\widehat{K_a}H_{a,b}\widehat{K_b}^{-1}$.

TODO

## Running Inference:

From the root directory, i.e. in the `ThermalFaceID_dist` directory, run `python main.py`. Running this script will execute the final product, which is the basic visual interface running our model inference.
   - There are several input arguments you can use to accomodate differences in computer set up. The following arguments are:
     - `-webcam_id` (`type=int, default=0`): Webcam ID if default detected webcam is not Logitech cam.
     - `-rotation` (`type=int, default=90`): Rotation for webcam if needed. accepted values are -90,90,180
     - `-height_ratio` (`type=float, default=0.75`): Minimum height ratio of frame for face to occupy.
     - `-face_confidence` (`type=int, default=0.7`): Facial landmark detection confidence threshold (MediaPipe parameter)
     - `-liveness_threshold` (`type=float, default=0.04`): liveness threshold.
     - `-heat_threshold` (`type=float, default=1.5`): thermal face variation threshold' options.
   - The interface is simple. RED = ANTI-SPOOF. GREEN = REAL FACE. If there is no box, either the face is too far, there is more than one face, or no face is detected. For detected faces, landmarks will be also drawn.
   - If you want to quit and exit, press `q`.

<p align="center">
    <img src="pictures\mainpreview.png" alt="mainpreview" width="25%">
</p>
<p align="center">
    <i>Ref 6: Visual interface from main.py script.</i>
</p>

The default arguments are loaded in through the `config.toml` file. This is because there are several common hyperparameter arguments that can be tuned, and have overlapped usage across multiple scripts. For convenience, if you want to change any of the hyperparameters, you can either do it manually via the argument commads within the command-line terminal, or change the .toml file. The hyperparameters for tuning the model, camera, etc. are under the `"tool"` key.


## Miscellaneous Analytics:

In the `/bin` folder, there are two scripts that are used for exploratory purposes as we continue to develop ideas for improving the overall framework. Keep in mind that these scripts are to be run **only after set up is completed**. 

`temp_tracker.py` is a script that displays the same visual interface as `main.py` but includes another window displaying the live temperature reading of a horizontal cut of the face from the nose. The purpose of this was to observe the difference in temperature patterns on horizontal cuts from the nose from real/fake faces.

`wavelength_test.py` is a script that displays the same visual interface as `main.py` but includes another window displaying the attempted live tracking of human 'pulses' from the detected face's forehead. The purpose of this was to see if there could be a detected pulse in temperature change that was different in real faces from fake faces. This idea was inspired by [Yu et al., 2021](https://www.techscience.com/cmc/v68n3/42515/html).

Both scripts have the optional input arguments `-webcam_id`, `-rotation`, `-confidence`, `-height_ratio` that are the same as the input arguments from main.py.


## Analysis and Results:
**[TO DO]**

## Contact:
If there are any questions, feel free to contact me at takao.kakegawa@meridianinno.com
