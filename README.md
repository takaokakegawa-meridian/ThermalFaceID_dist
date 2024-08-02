# Introduction:
This directory includes the work of the second iteration of the Depth-Based method inspired by Liu et al., 2017 [(PDF here)](https://cvlab.cse.msu.edu/pdfs/FaceAntiSpoofingUsingPatchandDepthBasedCNNs.pdf) that utilises fully convoluted networks to generate estimated thermal mappings, starting with a visual RGB image.

Building upon the first iteration, the second version of our facial anti-spoofing has diverged from using the Yolo framework, and instead utilises MediaPipe (Google API), which offers landmark detection with a model that is both memory/computationally more favourable.

# Set Up Guide:
### Hardware:
In order to use this, there are a few things to set up. Firstly, you will need the following hardware:
- Logitech C270 HD Webcam
- Panther 45
- SenXor MI48 compatbile with Panther
- Camera stand/mount
- 3D printed board for this set up (provided file)

For this set up, we have generated a custom board to mount both the visual/thermal cameras with their boards. This board can be 3D-printed using the file: **[ENTER FILENAME HERE]**. The board is to be mounted on the camera stand.

This is an example of an ideal/favorable set up:

<div style="text-align:center;">
    <img src="pictures/hardwaresetup.JPG" alt="Hardware Setup" width="30%">
</div>

### Software Environment:
This will be ideal to set up with Python 3.11. Any version of Python 3.10+ should suffice. Additionally, in a virtual environment, install all the necessary packages and libraries by running `pip install -r requirements.txt` from the root directory. You may be prompted to log in to GitHub to install pysenxor. 

### Homography calibration:
Next, you will need to calibrate the visual and thermal cameras to be aligned homographically. This can be completed with the scripts provided in `homography_alignment`. This folder contains the scripts `pair_record.py`, `pair_stream_calculate.py`, and `verify.py` that will be used in that order. From the root directory, i.e. in the `ThermalFaceID_dist` directory, do the following:

1. Run `python pair_record.py`. Running this script will open up two CV windows; one streaming the webcam, and one streaming Panther's camera.
   - There are several input arguments you can use to accomodate differences in computer set up. The following arguments are:
     - `-webcam_id` (`type=int, default=0`): Webcam ID if default detected webcam is not Logitech cam.
     - `-rotation` (`type=int, default=90`): Rotation for webcam if needed. accepted values are -90,90,180
     - `-colormap` (`type=str, default='inferno'`): Thermal image colormap choice. Options correspond to pysenxor options.

   - There are four keyboard commands: `p` to pause the streams. `r` to resume the streams. `q` to quit and exit. `m` to start recording and saving images.
2. It is recommended that you give 4-5 seconds for Panther to calibrate, and come up close/stay very still in front of both cameras so that your entire face covers majority of the frame. When ready, press `m` and this will start recording the simultaneous streams. It will save the thermal/visual image pairs into a folder called `homography_sampleset` and in separate `thermal` and `visual` folders. The pairs are matched by having the same filename. When you have recorded enough (~2-3 seconds), press `q` to quit.
3. Next create a copy of `homography_coord_template.csv` in the same `homography_alignment` folder, and name it something different. Preferably, `homography_coord.csv`. If you name it something else, that is ok. Keep in mind you will need to use input arguments in latter scripts if you name it something else.
4. Run `python pair_stream_calculate.py`. Running this script will open a matplotlib window of the visual/thermal image pairs. 
