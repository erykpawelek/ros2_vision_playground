# ROS 2 Vision Playground 

A modular, high-performance computer vision node designed for **Raspberry Pi 5** running **Ubuntu 24.04** (Noble) and **ROS 2 Jazzy**.

This project serves as a development platform for experimenting with real-time perception algorithms, sensor integration, and distributed system architecture. The goal is to create a robust "visual cortex" capable of analyzing the environment and generating high-level control signals (e.g., error vectors) for autonomous robot navigation.

## Key Features

* **Dispatcher Architecture:** Enables dynamic run-time switching between processing algorithms (Classical CV vs. AI) via ROS parameters, ensuring zero downtime.
* **Color Object Tracking:** Implements HSV thresholding, contour analysis, and moment-based centroid calculation for high-speed tracking (60+ FPS).
* **AI Hand Tracking:** Integrates Google MediaPipe for robust hand landmark detection using an asynchronous pipeline.
* **Bandwidth Optimization:** Utilizes `sensor_msgs/CompressedImage` for transport to minimize Wi-Fi saturation and latency during remote debugging.
* **Dynamic Reconfiguration:** Key parameters (e.g., HSV thresholds) can be tuned dynamically at runtime.
* **Portable Design:** Uses dynamic path resolution (`ament_index_python`) for model files, ensuring the package runs on any machine without hardcoded paths.

## Engineering Challenge: Handling Asynchronous Inference Latency

A critical challenge encountered during the development of the AI module was managing the discrepancy between the high data rate of the sensor and the inference speed of the CPU.

### The Problem: Buffer Bloat
The camera hardware (IMX708) operates at **30 FPS** (generating data every ~33ms). However, complex neural network inference on the CPU takes approximately **90-100ms** per frame (~10 FPS). 

In a standard asynchronous callback structure, the ROS 2 message queue filled rapidly with unprocessed frames. This "buffer bloat" resulted in a constantly increasing system latency that peaked at **~7 seconds**, making real-time control impossible.

### The Solution: Application-Level Flow Control
To resolve this, a **Frame Decimation (Skipping)** logic was implemented at the ingress point of the node.

* **Throughput Matching:** The system actively monitors the input frame sequence.
* **Deterministic Dropping:** The algorithm systematically discards frames that exceed the processing capacity of the inference engine. For a 30 FPS input and ~10 FPS inference capability, the node processes only every **3rd** frame.
* **Results:**
    * **Zero-Queue Operation:** Prevents the accumulation of stale data in input buffers.
    * **Latency Reduction:** Reduced end-to-end system latency from **~7000ms to <200ms**.
    * **Resource Management:** Allows the CPU to focus solely on the most recent visual data, ensuring control signals reflect the current state of reality.

## Hardware & Prerequisites

* **Hardware:** Raspberry Pi 5 (8GB RAM) + Raspberry Pi Camera Module 3 (CSI).
* **OS:** Ubuntu 24.04 LTS (Noble Numbat).
* **ROS Distro:** ROS 2 Jazzy Jalisco.

> **Important:** Standard Ubuntu drivers do not support the RPi Camera Module 3 correctly out of the box. You MUST build the Raspberry Pi fork of `libcamera` from source. 
> See my detailed guide here: **[Raspberry Pi 5 + Camera Module 3 Setup Guide](https://github.com/erykpawelek/libcamera_ros2_setup)**

## Dependencies

Ensure you have the following installed on your Raspberry Pi:

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport-plugins python3-opencv
# Note: It is recommended to use a virtual environment for Python packages
pip3 install mediapipe
```
## Architecture Notes & Experimental Branch

During development, two different architectural approaches were tested to handle the AI inference load on the Raspberry Pi CPU.

#### 1.Synchronous Mode (Production - `main`)
* Current Default.
* Uses `RunningMode.VIDEO`.
* Pros: Zero latency, predictable behavior. Best for real-time analysis.

#### 2.Asynchronous Mode (Experimental)
* Uses RunningMode.LIVE_STREAM with a custom frame skipping algorithm.
* Pros: Decouples camera acquisition from processing.
* Cons: Introduced significant latency (~7s) on this hardware configuration due to internal buffering limitations (before frame skipping was implemented).