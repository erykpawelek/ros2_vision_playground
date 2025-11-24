# ROS 2 Vision Playground 

A custom ROS 2 package for real-time computer vision processing, designed for **Raspberry Pi 5** running **Ubuntu 24.04 (Noble)** and **ROS 2 Jazzy**.

This node subscribes to a camera stream, performs color detection using HSV thresholding, identifies object contours, calculates the centroid (center of mass), and visualizes the tracking error for potential robot control.

## ✨ Features

* **Efficient Data Handling:** Uses `CompressedImage` transport to minimize Wi-Fi bandwidth usage and reduce latency.
* **Color Tracking:** HSV-based color filtration (default: blue/cyan range).
* **Advanced Processing:**
    * Contour detection to filter out noise.
    * Centroid calculation using Image Moments.
    * Visualizes the target center and offset from the screen center.
* **Configurable:** Key parameters (HSV thresholds) can be tuned dynamically at runtime.
* **Output:** Publishes the processed debug image with overlays (bounding box, center point, text stats).

## Hardware & Prerequisites

* **Hardware:** Raspberry Pi 5 + Raspberry Pi Camera Module 3 (CSI).
* **OS:** Ubuntu 24.04 LTS.
* **ROS Distro:** ROS 2 Jazzy Jalisco.

> **Important:** Before using this package, ensure your Raspberry Pi camera drivers are correctly installed and configured. See my detailed guide here:
> **[Raspberry Pi 5 + Camera Module 3 Setup Guide](https://github.com/erykpawelek/libcamera_ros2_setup)**

## Dependencies

Ensure you have the following installed on your Raspberry Pi:

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport-plugins python3-opencv