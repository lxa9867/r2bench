import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np

from PIL import Image


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import random

warnings.simplefilter("ignore", UserWarning)

def depth_add_gaussian_noise(x, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    mean = np.mean(x) * c
    std = np.std(x) * c
    noise = np.random.normal(mean, std, x.shape)
    noise = noise.reshape(x.shape).astype('uint16')
    noisy_image = x + noise
    return noisy_image

def depth_add_edge_erosion(x, severity=1):
    c = [(0.015, 3), (0.020, 3), (0.025, 3), (0.03, 3), (0.035, 3)][severity - 1]
    random_rate = c[0]
    patch_len = c[1]
    scaled_x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(scaled_x, 20, 50)
    gauss = np.full(x.shape, 0,dtype=np.uint16)
    edge_pixel = []
    # Create a mask where edges are 1
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] > 0:
                edge_pixel.append([i,j])
    edge_pixel_num = len(edge_pixel)
    erosion_edge = random.sample(edge_pixel, int(edge_pixel_num * random_rate))
    for pixel in erosion_edge:
        edges[pixel[0] - patch_len:pixel[0] + patch_len, pixel[1] - patch_len:pixel[1] + patch_len] = 1
    edge_mask = edges > 0
    # Apply Gaussian noise only to the edge pixels
    noisy_image = np.copy(x)
    noisy_image[edge_mask] = noisy_image[edge_mask] * gauss[edge_mask]

    return noisy_image

def depth_add_random_mask(x, severity=1):
    c = [5, 7, 9, 11, 13][severity - 1]
    num_rectangles = c
    scale = 0.1
    patch_w = int(x.shape[0] * scale)
    patch_h = int(x.shape[1] * scale)
    mask = np.zeros(x.shape, dtype=np.uint16)
    start_point = []
    sampled_num = 0
    while True:
        x1, y1 = np.random.randint(0, x.shape[0] - patch_w), np.random.randint(0, x.shape[1] - patch_h)
        if len(start_point) == 0:
            start_point.append((x1,y1))
        else:
            for point in start_point:
                if np.abs(point[0] - x1) < patch_w or np.abs(point[1] - y1) < patch_h:
                    continue
                else:
                    start_point.append((x1,y1))
        x2, y2 = x1 + patch_w, y1 + patch_h
        mask[x1:x2,y1:y2] = 1
        sampled_num += 1
        if sampled_num == num_rectangles:
            break
    gauss = np.full(x.shape, 0, dtype=np.uint16)
    noisy_image = np.copy(x)
    mask = mask < 1
    gauss[mask] = 1
    noisy_image = noisy_image * gauss
    return noisy_image
def depth_add_fixed_mask(x, severity=1):
    c = [5, 7, 9, 11, 13][severity - 1]
    scale = 0.1
    patch_w = int(x.shape[0] * scale)
    patch_h = int(x.shape[1] * scale)
    mask = np.zeros(x.shape, dtype=np.uint16)
    start_point = [(1,1),(3,1), (5,1), (7,1),(1,3),(1,5),(1,7),(3,3),(5,5), (9,9),(9,1),(1,9),(7,7)][:c]
    for i in range(c):
        x1, y1 = (start_point[i][0]-1) * patch_w, (start_point[i][1]-1) * patch_h
        x2, y2 = x1 + patch_w, y1 + patch_h
        mask[x1:x2, y1:y2] = 1
    gauss = np.full(x.shape, 0, dtype=np.uint16)
    noisy_image = np.copy(x)
    mask = mask < 1
    gauss[mask] = 1
    noisy_image = noisy_image * gauss
    return noisy_image
def depth_range(x, severity=1):
    c = [(0.2, 3), (0.3, 3.2), (0.4, 3.4), (0.5, 3.6), (0.6, 3.8)][severity - 1]
    min = c[0]
    max = c[1]
    mask_sign = 0
    filtered_image = np.copy(x)
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            if filtered_image[i][j] > max or filtered_image[i][j] < min:
                filtered_image[i][j] = mask_sign
    return filtered_image




