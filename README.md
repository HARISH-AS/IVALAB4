Overview
This project demonstrates the implementation of a video analytics system using Python and OpenCV. The system performs three main tasks:

Scene Cut Detection using Structural Similarity Index (SSIM)
Object Segmentation using color-based and edge-based segmentation techniques
Object Tracking using centroid detection and motion tracking.
These tasks are crucial for applications like video surveillance, sports analytics, and content moderation, where video streams need to be analyzed automatically to extract meaningful insights.

Table of Contents
Objective
Problem Statement
Methodology
Algorithms
Results and Discussion
Installation and Setup
How to Run the Code
References
Objective
The objective of this lab assignment is to develop a Python-based system capable of:

Detecting scene changes in a video using SSIM.
Segmenting objects from video frames using both color-based and edge-based methods.
Tracking the movement of segmented objects across multiple frames.
This system is aimed at automating the process of analyzing video streams, providing efficient solutions for applications such as surveillance and video summarization.

Problem Statement
With the rapid increase in video data generation, manual analysis of video streams is both time-consuming and error-prone. Automating video analysis involves addressing several challenges:

Scene Cut Detection: Identifying where and when scene transitions occur (either through abrupt cuts or gradual changes) is essential for structuring video content.
Object Segmentation: Extracting meaningful objects from the video requires accurate segmentation methods, capable of handling variations in lighting, shape, and motion.
Object Tracking: Tracking the movement of objects across video frames is necessary for detecting behavior patterns, monitoring activity, and identifying anomalies.
Methodology
The video analytics system developed in this project consists of the following steps:

Video Loading and Frame Extraction:

The input video is loaded using OpenCV, and each frame is extracted for further processing.
Object Segmentation:

Color-Based Segmentation: Using HSV color space, objects of specific colors (like red or green) are segmented from each frame.
Edge Detection (Sobel): The Sobel operator is applied to detect edges within the frames, highlighting object contours.
Scene Cut Detection:

The Structural Similarity Index (SSIM) is used to compare consecutive frames, detecting abrupt or gradual scene changes based on SSIM scores.
Object Tracking:

The centroids of segmented objects are calculated and tracked across frames to monitor their motion paths.

Algorithms
Scene Cut Detection (SSIM):
Convert video frames to grayscale.
Calculate SSIM between consecutive frames.
Use thresholds to classify SSIM scores as hard or soft scene transitions.
Object Segmentation:
Color-Based Segmentation: Convert frames to HSV and segment objects within specific color ranges.
Edge Detection (Sobel): Apply the Sobel operator to highlight object edges by calculating horizontal and vertical gradients.
Object Tracking:
Detect object contours.
Calculate object centroids.
Track centroid positions across frames to visualize movement.
Results and Discussion
Scene Cut Detection:

Detected both hard and soft scene cuts using SSIM. Sharp drops in SSIM values indicated hard cuts, while gradual transitions were captured as soft cuts.
Object Segmentation:

Successfully segmented objects based on color and detected edges using Sobel. Color segmentation was effective for objects with distinct colors, while Sobel highlighted object contours.
Object Tracking:

Centroid-based tracking was applied to moving objects, and their trajectories were visualized. This was particularly useful in tracking motion across frames.
