#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  14 18:01:41 2023 

@author: sajjad
"""

# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to load and preprocess the image
def ReadGrayLevelImage(F_name):
    """
    Reads an image, converts it to grayscale, and applies smoothing to reduce noise.
    Parameters:
        F_name (str): Path to the image file.
    Returns:
        np.ndarray: Preprocessed grayscale image.
    """
    if not os.path.exists(F_name):
        raise FileNotFoundError(f"Image file '{F_name}' not found.")
    I = cv2.imread(F_name)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    I = cv2.GaussianBlur(I, (11, 11), 0)    # Apply Gaussian blur to reduce noise
    return I

# Function to compute Otsu's threshold
def FindOtsuThreshold(I):
    """
    Computes the optimal threshold for binary segmentation using Otsu's method.
    Parameters:
        I (np.ndarray): Grayscale image.
    Returns:
        int: Optimal threshold value.
    """
    _, T_gray = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return T_gray

# Function to perform binary thresholding
def ThresholdImage(I, T_gray):
    """
    Converts the image into a binary format using a given threshold.
    Parameters:
        I (np.ndarray): Grayscale image.
        T_gray (int): Threshold value.
    Returns:
        np.ndarray: Binary thresholded image.
    """
    _, bw = cv2.threshold(I, T_gray, 255, cv2.THRESH_BINARY_INV)
    return bw

# Function to compute the distance transform
def OuterDistanceTransform(bw):
    """
    Computes the distance transform of a binary image.

    Parameters:
        bw (np.ndarray): Binary image.

    Returns:
        np.ndarray: Distance transform image.
    """
    return cv2.distanceTransform(bw, cv2.DIST_L2, 5)

# Function to identify seeds for segmentation
def IdentifySeeds(D, T_dist):
    """
    Identifies seed regions in the distance transform based on a distance threshold.
    Parameters:
        D (np.ndarray): Distance transform image.
        T_dist (float): Distance threshold.
    Returns:
        np.ndarray: Labeled seed regions.
    """
    seeds = (D > T_dist).astype(np.uint8) * 255
    num_components, labels = cv2.connectedComponents(seeds)
    return labels

# Function to grow seeds for segmentation
def GrowSeeds(S_regions, D):
    """
    Expands seed regions to cover the entire objects.
    Parameters:
        S_regions (np.ndarray): Initial seed regions.
        D (np.ndarray): Distance transform image.
    Returns:
        np.ndarray: Grown seed regions (segmentation map).
    """
    markers = cv2.watershed(cv2.cvtColor((D * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), S_regions)
    return markers

# Function to visualize segmentation results
def DisplayResults(original, thresholded, distance, seeds, segmented):
    """
    Displays the original, thresholded, distance transform, and segmented images.
    Parameters:
        original (np.ndarray): Original grayscale image.
        thresholded (np.ndarray): Binary thresholded image.
        distance (np.ndarray): Distance transform image.
        seeds (np.ndarray): Seed regions.
        segmented (np.ndarray): Segmentation map.
    """
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1), plt.imshow(original, cmap='gray'), plt.title("Original Image")
    plt.subplot(2, 3, 2), plt.imshow(thresholded, cmap='gray'), plt.title("Thresholded Image")
    plt.subplot(2, 3, 3), plt.imshow(distance, cmap='jet'), plt.title("Distance Transform")
    plt.subplot(2, 3, 4), plt.imshow(seeds, cmap='nipy_spectral'), plt.title("Seed Regions")
    plt.subplot(2, 3, 5), plt.imshow(segmented, cmap='nipy_spectral'), plt.title("Segmented Image")
    plt.tight_layout()
    plt.show()

# Main function to segment objects in the image
def SegmentObjects(F_name, T_dist):
    """
    Segments objects in a grayscale image using Otsu's thresholding and distance transform.
    Parameters:
        F_name (str): Path to the image file.
        T_dist (float): Distance threshold for seed identification.
    Returns:
        None
    """
    I = ReadGrayLevelImage(F_name)
    T_gray = FindOtsuThreshold(I)
    bw = ThresholdImage(I, T_gray)
    D = OuterDistanceTransform(bw)
    seeds = IdentifySeeds(D, T_dist)
    segmented = GrowSeeds(seeds, D)
    DisplayResults(I, bw, D, seeds, segmented)

# Example usage
if __name__ == "__main__":
    image_path = "./sample1.png"
    distance_threshold = 50
    SegmentObjects(image_path, distance_threshold)
