# Object_detection_using_YOLO

This project implements real-time person detection and distance estimation using the YOLO model and OpenCV. It also provides audio alerts using a text-to-speech engine when a person is detected within a specified distance.


## Features
• Real-time person detection using YOLO  
• Distance estimation based on object size  
• Audio alerts using pyttsx3  
• Multi-threaded speech processing to prevent blocking  
• Works with a webcam for live detection  

## Requirements
Ensure you have the following dependencies installed:
```
pip install ultralytics opencv-python pyttsx3
```

## Usage

• Run the script to start the person detection system:

```
python person_detection.py
```

• Press ```q``` to exit the detection window.


## How It Works

1. YOLO Model: The script loads a YOLO model for detecting persons in the webcam feed.

2. Distance Estimation: Uses the known height of a person and the bounding box height in pixels to estimate distance.

3. Audio Alerts: A text-to-speech engine provides spoken alerts when a person is detected within a critical range.

4. Multi-Threading: Speech processing runs on a separate thread to prevent performance bottlene

## Configuration

Modify ```constant.py``` to adjust parameters such as:

• ```MODEL_PATH:``` Path to the YOLO model.

• ```FRAME_WIDTH, FRAME_HEIGHT:``` Webcam resolution.

• ```KNOWN_HEIGHT:``` Reference height of a person for distance estimation.

• ```REFERENCE_DISTANCE:``` Distance calibration for focal length calculation.

## Acknowledgments

• YOLO for object detection: Ultralytics

• OpenCV for image processing

• pyttsx3 for text-to-speech conversion

## License

This project is licensed under the MIT License.