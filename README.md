# ROS 2 Vision Playground (Async Experimental Branch) 

This branch implements a computer vision node for **Raspberry Pi 5** using an **Asynchronous Inference Architecture**. It decouples the camera frame acquisition from the neural network processing to potentially maximize throughput, using a smart frame-skipping mechanism to manage latency.

## Architecture: Asynchronous Dispatcher

In this version, the node operates in a non-blocking manner:
1.  **Input:** Receives camera frames continuously (up to 30 FPS).
2.  **Dispatch:** Sends frames to the MediaPipe background thread via `detect_async`.
3.  **Callback:** Receives results in a separate callback function once inference is complete.
4.  **Output:** Publishes processed images as soon as they are ready.

### The "Lag" Challenge & Solution
Standard asynchronous processing on limited hardware (CPU only) leads to a queue buildup, resulting in significant latency (up to 7 seconds).

**Solution Implemented: Frame Skipping Gatekeeper**
To synchronize the high-speed camera (30 FPS) with the slower CPU inference (~10-11 FPS), a strict gatekeeper logic is applied:
* The node counts every incoming frame.
* Only **every 3th frame** is passed to the inference engine.
* All other frames are dropped immediately to prevent buffer bloat.
* **Result:** Smooth operation with ~10 FPS output and manageable latency.

## Features (Branch Specific)

* **Mode:** `neural_net_mediapipe` uses `RunningMode.LIVE_STREAM` (Async).
* **Callback System:** Implements `neural_net_mediapipe_callback` to handle inference results from a separate thread.
* **Visualization:** Draws the *latest available* inference result on the *current* video frame (may show slight misalignment during fast motion due to async nature).