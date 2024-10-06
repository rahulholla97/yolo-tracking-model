# Object Tracking Model

This project implements a system for tracking and classifying children and therapists in video footage using YOLOv8 object detection and custom classification models.

## Overview

The system uses two YOLO models:

1. A YOLOv8s model for person detection and tracking
2. A custom YOLOv8 model for classifying detected persons as either children or therapists

The main script processes video input, detects and tracks persons, and classifies each tracked individual over time.

## Prerequisites

- Python 3.9
- OpenCV [`cv2`]
- NumPy
- Ultralytics YOLO

## Installation

1. Install the required packages:

   ```python
   pip install opencv-python numpy ultralytics
   ```

2. Download the necessary YOLO models:
   - YOLOv8s model for person detection
   - Custom YOLOv8 model for child/therapist classification

## Usage

1. Place your input video in the project directory and name it `video.mp4`.
2. Run the main script:

   ```python
   python inference.py
   ```

3. The script will process the video and output:
   - A real-time display of the annotated video
   - An output video file named `predicted_output.mp4`

## Test Video

You can download the test video used for this project from the following link:
[Test Video Link](https://www.youtube.com/watch?v=fEEelCgBkWA)

## Predicted Output Video

The predicted output video, showing the tracking and classification results, can be viewed here:
[Predicted Output Video Link](https://drive.google.com/file/d/13TgLkoVrjlRYVjaiTQQhlhr842e9-yPb/view?usp=sharing)

## Model Training

The custom classification model was trained using YOLOv8 and a dataset from Roboflow. For more details on the training process, refer to `training.py`.

## Key Features

- Person detection and tracking using YOLOv8 and ByteTrack
- Custom classification of tracked individuals as children or therapists
- Temporal smoothing of classifications to improve accuracy
- Real-time visualization of tracking and classification results
- Output video generation with annotations

## Configuration

You can adjust the following parameters in the script:

- `time_threshold`: Time interval for updating classifications (default: 3 seconds)
- `confidence_threshold`: Confidence threshold for high accuracy classifications
- `outperform_threshold`: Number of intervals a class needs to outperform the current best class

## Output

The script generates an annotated video showing:

- Bounding boxes around detected persons
- Track IDs for each person
- Current classification (child or therapist) for each tracked individual

## Limitations

- The system assumes that the input video contains only children and therapists.
- Classification accuracy may vary depending on the quality of the input video and the training data used for the custom model.

## Future Improvements

- Implement multi-class classification for more detailed analysis
- Optimize performance for real-time processing of high-resolution videos
- Develop a user interface for easier configuration and result analysis

## Acknowledgements

- Ultralytics for the YOLOv8 implementation
- Roboflow for providing the dataset used in training the custom model
