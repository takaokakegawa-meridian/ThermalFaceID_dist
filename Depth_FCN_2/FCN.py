"""
File: FCN.py
Author: Takao Kakegawa
Date: 2024
Description: Script with all classes and intermediate functions to handle train/val/test
             data. Additionally, there the training script that can be used for retraining
             purposes.
"""

import copy
import json
import os
from argparse import Namespace
from typing import List, Any, Dict

import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


## Depth FCN independent to input shape (3,H,W). 
class DepthBasedFCN(nn.Module):
    """class for Depth-based Fully Convolutional Network (FCN) model object.
    """
    def __init__(self, input_channels: int):
        """__init__ method to establish the network parameters and framework shape.
        Args:
            input_channels (int): number of input channels. A 3-channel image
                                   for example would require setting input_channels = 3
        Returns:
            None
        """
        super(DepthBasedFCN, self).__init__()
        #### Upsampling stage
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(256, 160, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #### Downsampling. LeakyReLU across each convolutional layer
        self.conv3_1 = nn.Conv2d(160, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.convt_32 = nn.Conv2d(128, 128, kernel_size=6, padding=5)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv4_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        self.convt_42 = nn.Conv2d(128, 128, kernel_size=6, padding=5)
        self.relu4 = nn.LeakyReLU(0.1)
        
        self.conv5_1 = nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU(0.1)
        self.convt_52 = nn.Conv2d(160, 160, kernel_size=6, padding=5)
        self.relu6 = nn.LeakyReLU(0.1)
        
        self.conv6_1 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.LeakyReLU(0.1)
        self.convt_62 = nn.Conv2d(320, 320, kernel_size=6, padding=5)
        self.relu8 = nn.LeakyReLU(0.1)

        self.conv7_1 = nn.Conv2d(320, 1, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function for singular pass with input tensor in FCN.
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu1(x)
        x = self.convt_32(x)
        x = self.relu2(x)

        x = self.conv4_1(x)
        x = self.relu3(x)
        x = self.convt_42(x)
        x = self.relu4(x)

        x = self.conv5_1(x)
        x = self.relu5(x)
        x = self.convt_52(x)
        x = self.relu6(x)
        
        x = self.conv6_1(x)
        x = self.relu7(x)
        x = self.convt_62(x)
        x = self.relu8(x)

        x = self.conv7_1(x)
        x = self.relu9(x)

        return x


class FCN_Dataset(Dataset):
    """Custom torch.Dataset class to acccomodate raw data collected for DepthBasedFCN module.
    """

    def __init__(self, thermal_json: str, rgb_root: str, landmark_coords: str):
        """__init__ method to build dataset to use with DepthBasedFCN module.
        Args:
            thermal_json (str): Path string of JSON file for thermal frame data. In {id:frame} format.
            rgb_root (str): Path string of root where of HSV+YCrCb converted (visual) images. Filenames must match thermal ids.
            landmark_coords (str): Path string of JSON file for landmark coordinates data. In {id:coordinates} format.

            Example:    thermal_json = r"thermal_actual_values.json"
                        rgb_root = r"RGB-transformed2-faces-128x128"
                        landmark_coords = r"landmarkcoords.json"
        
        Returns:
            None
        """
        try:
            # directory for HSV+YCrCb input images.
            self.rgb_root = rgb_root
            # json containing the landmark actual 'normalized' thermal values.
            with open(thermal_json, "r") as f:
                self.thermal_files = json.load(f)
            # json containing the landmark coordinates for rgb input images.
            with open(landmark_coords, "r") as f:
                self.landmark_coords = json.load(f)
            
            self.file_names = sorted(list(set(os.listdir(rgb_root))))
            assert(len(set(self.landmark_coords.keys())) == len(self.file_names)), "mismatch in number of file names."

        except FileNotFoundError:
            print("one or more paths not found")


    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.Tensor]:
        """__getitem__ method to get desired inputs for model training and evaluation.
        Args:
            idx (int): index of data to get.
        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.Tensor]: returns actual thermal landmark 
            values, 3-channel visual image, the visual image dimensions, and the landmarks coordinates for the
            visual image.
        """
        thermal_vals = np.array(self.thermal_files[self.file_names[idx]])
        rgb_img = cv.imread(os.path.join(self.rgb_root, self.file_names[idx]))
        rgb_shape = rgb_img.shape[:2]
        thermal_vals = torch.from_numpy(thermal_vals).float()
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        landmarks = np.array(self.landmark_coords[self.file_names[idx]]["rgb"])
        landmarks = torch.from_numpy(landmarks).int()
        assert landmarks.shape[0] == thermal_vals.shape[0], "mismatch with number of landmarks"
        return thermal_vals, rgb_img, rgb_shape, landmarks


def my_collate(batch: List[tuple]) -> list[torch.Tensor, List, List, List]:
    """ Collate a batch of data into separate lists for thermal values, RGB images, RGB shapes, and landmarks.
    Args:
        batch (List[tuple]): A list of tuples where each tuple contains thermal values, RGB images, RGB shapes, and landmarks.

    Returns:
        list[torch.Tensor, List, List, List]: A list containing the following:
            - thermal_val (torch.Tensor): Stacked thermal values from the input batch.
            - rgb_img (List): List of RGB images extracted from the batch.
            - rgb_shape (List): List of RGB shapes extracted from the batch.
            - landmarks (List): List of landmarks extracted from the batch.
    """
    thermal_val = torch.stack([item[0] for item in batch])
    rgb_img = [item[1] for item in batch]
    rgb_shape = [item[2] for item in batch]
    landmarks = [item[3] for item in batch]
    return [thermal_val, rgb_img, rgb_shape, landmarks]


def pad_tensors(tensor_lst: List[torch.Tensor]) -> torch.Tensor:
    """function that pads a list of 2D tensors to all have the same size for batch-wise loss calculation
    Args:
        tensor_lst (List[torch.Tensor]): list of tensors from a batch
    Returns:
        torch.Tensor: tensor of padded tensors, all stacked together.
    """
    max_rows = max(tensor.size(0) for tensor in tensor_lst)
    max_cols = max(tensor.size(1) for tensor in tensor_lst)
    return torch.stack([F.pad(t, (0, max_cols-t.size(1), 0, max_rows-t.size(0))) for t in tensor_lst])


def train_FCN(args: Namespace, model: DepthBasedFCN, device: str,
              traindataset: FCN_Dataset, valdataset: FCN_Dataset) -> tuple[Dict[str, Any], List[float], List[float]]:
    """ Train a Fully Convolutional Network (FCN) model using the specified datasets, hyperparameter
        arguments, device, and model framework.
    Args:
        args (Namespace): Parsed arguments containing hyperparameter configurations.
        model (DepthBasedFCN): The FCN model to be trained.
        device (str): The device on which to train the model (e.g., 'cpu', 'cuda').
        traindataset (FCN_Dataset): Dataset used for training.
        valdataset (FCN_Dataset): Dataset used for validation.
    Returns:
        Tuple[Dict[str, Any], List[float], List[float]]: A tuple containing the best model state as a dictionary,
        a list of training losses, and a list of validation losses.
    """
    best_loss, best_model_state = float('inf'), None
    train_losses, val_losses = [], []
    train_loader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("START TRAINING...")
    for epoch in range(1, args.epochs + 1):
        print(f"EPOCH {epoch} NOW ...")
        train_loss = 0.0
        model.train()

        #### Training process:
        for _, (thermal_vals, rgb_imgs, rgb_shapes, landmarks) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = []
            for rgb_img, rgb_shape, landmark in zip(rgb_imgs, rgb_shapes, landmarks):
                output = model(rgb_img.to(device)).unsqueeze(dim=0)
                output_resized = F.interpolate(output, size=rgb_shape,
                                                mode='bicubic').squeeze()
                x_coords, y_coords = landmark[:, 0], landmark[:, 1]
                output_vals = output_resized[y_coords, x_coords]
                outputs.append(output_vals)
            padded_output = torch.stack(outputs).to(device)
            padded_thermal = thermal_vals.to(device)
            loss = F.mse_loss(padded_output, padded_thermal)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"EPOCH {epoch} TRAIN loss: {train_loss}")

        #### Validation process at regular intervals:
        if epoch % args.log_interval == 0:
            print(f"[EPOCH {epoch}] VALIDATION CHECK:")
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for _, (thermal_vals, rgb_imgs, rgb_shapes, landmarks) in enumerate(val_loader):
                    outputs = []
                    for rgb_img, rgb_shape, landmark in zip(rgb_imgs, rgb_shapes, landmarks):
                        output = model(rgb_img.to(device)).unsqueeze(dim=0)
                        output_resized = F.interpolate(output, size=rgb_shape,
                                                        mode='bicubic').squeeze()
                        x_coords, y_coords = landmark[:, 0], landmark[:, 1]
                        output_vals = output_resized[y_coords, x_coords]
                        outputs.append(output_vals)
                    padded_output = torch.stack(outputs).to(device)
                    padded_thermal = thermal_vals.to(device)
                    loss = F.mse_loss(padded_output, padded_thermal)
                    val_loss += loss.item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'[EPOCH {epoch}] TRAIN LOSS: {train_loss}, VAL LOSS: {val_loss}')
            if val_loss < best_loss:
                print(f'[IMPROVEMENT]: OLD LOSS: {best_loss}, NEW LOSS: {val_loss}')
                best_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
        print("----"*25)
    print("Done Training")
    return best_model_state, train_losses, val_losses
