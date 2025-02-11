# Seed_Growth_Segmentation
This Python script implements a comprehensive image segmentation pipeline aimed at isolating objects in grayscale images, such as coins in a photograph.

This code was part of a programming competition at Koc University (KUIS AI Center).

This code implements an image segmentation pipeline that utilizes a combination of thresholding, distance transforms, and seed-based region growing to identify and delineate objects in grayscale images. The process begins with preprocessing, where an input image is converted to grayscale and smoothed to reduce noise. Using Otsu's method—a thresholding technique that determines an optimal global threshold by maximizing the variance between two classes—the image is binarized to separate foreground objects from the background. Subsequently, a distance transform computes the shortest distance of each foreground pixel to the nearest background pixel, creating a gradient representation of object proximity.

The segmentation process leverages seed-based region growing, a method that uses identified seed regions—specific pixels or clusters—as starting points. These seeds are identified from the distance transform using a user-defined threshold. The seeds are then expanded iteratively using a watershed algorithm, which assigns each pixel to the nearest seed based on a gradient-based flooding approach. This results in distinct segmentation maps that partition the image into meaningful regions corresponding to individual objects. Visualization functions are incorporated to present intermediate steps, such as the thresholded image, distance transform, and final segmentation, ensuring transparency in the pipeline.

This segmentation pipeline has diverse applications, particularly in fields requiring object identification and analysis. For instance, it can be employed in medical imaging to segment cells, organs, or tumors from medical scans, aiding diagnostics. In manufacturing, it can detect defect in industrial products or segment objects for automated assembly processes. The code's modularity and reliance on key computer vision techniques like Otsu's thresholding and seed-based region growth make it a versatile tool for both academic research and industrial applications.
