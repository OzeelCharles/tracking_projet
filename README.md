# CitySkyline Traffic Tracker | OpenCV & Clean Architecture

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?logo=pytest&logoColor=blue)

A pragmatic, lightweight, and highly structured object tracking and counting system built from scratch using traditional Computer Vision (OpenCV) and pure Python.

##  The Philosophy: Pragmatism over Hype

In modern Data Science, the default answer to object detection is often a heavy Deep Learning model (YOLO, CNNs, Mask R-CNN). However, for static-camera scenarios like traffic monitoring, traditional **Background Subtraction** combined with **Centroid Tracking** is exceptionally elegant, computationally cheap, and highly interpretable. 

This project was built to demonstrate that **Data Science is about understanding the data and choosing the right tool for the job**. It heavily focuses on **Software Craftsmanship**, separating pure mathematical logic from stateful tracking and visual rendering.

## Features

- **Stateless Detection:** robust background subtraction and morphological operations.
- **Stateful Centroid Tracking:** Euclidean distance-based object matching across frames.
- **Memory Leak Prevention (TTL):** A robust "deregister" garbage collector that cleans up lost or out-of-frame objects.
- **Line Crossing Logic:** Geometric calculations to detect when an object crosses a virtual counting line, avoiding duplicate counts.
- **Config-Driven:** No magic numbers. Everything is decoupled in a `config.py` and `.env` file.
- **100% Testable:** Pure logic is isolated from the OpenCV video stream, allowing for comprehensive Unit Testing.

## Architecture

The codebase follows SOLID principles and separates concerns into specific modules:

```text
city_tracker/
├── data/                    # Local video samples
├── src/
│   ├── main.py              # Orchestrator & Video Stream loop
│   ├── config.py            # Thresholds, distances, colors, configurations
│   ├── detector.py          # MotionDetector class (Stateless)
│   └── tracker.py           # CentroidTracker class (Stateful memory)
├── utils/
│   ├── geometry.py          # Pure math (distances, relative positions)
│   ├── image_processing.py  # Image transformations (blur, grayscale, etc.)
│   └── visualization.py     # Drawing bounding boxes and UIs
├── tests/
│   └── test_tracker.py      # Pytest unit tests for tracking & counting logic
├── .env                     # Network video stream URL (ignored in git)
├── requirements.txt         
└── README.md
