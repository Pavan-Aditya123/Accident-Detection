# AI Smart Accident Detection System

A YOLO26-based accident detection system for CCTV video analysis. Detects accidents and objects in traffic footage with bounding box annotations and confidence scores.

---

# Features

- YOLO26-based accident detection
- 10-class object detection (accidents, vehicles, pedestrians)
- Image and video inference support
- Annotated output with bounding boxes, class names, and confidence scores

---

# System Architecture

```
Image/Video Input
        ↓
YOLO26 Model
        ↓
Detection
        ↓
Class + Bounding Box + Confidence
        ↓
Annotated Output
```

---

# YOLO26 Classes

| Class |
|---|
| bike |
| bike_bike_accident |
| bike_object_accident |
| bike_person_accident |
| car |
| car_bike_accident |
| car_car_accident |
| car_object_accident |
| car_person_accident |
| person |

---

# Technologies Used

**Deep Learning:**
- YOLO26 (Ultralytics)

**Programming:**
- Python

**Libraries:**
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

**Hardware:**
- NVIDIA GPU with CUDA support (optional)

---

# Project Structure

```
AI-Smart-Accident-Detection-System/

├── inference/
│   └── accident_detection.py
├── training/
│   └── detection/
│       └── train_yolo26.py
├── datasets/
│   └── accident_detection/
│       ├── train/
│       ├── valid/
│       └── test/
├── results/
│   └── detection/
│       └── yolo26_baseline/
│           └── weights/
│               └── best.pt
├── requirements.txt
└── README.md
```

---

# Installation

```bash
cd AI-Smart-Accident-Detection-System

pip install -r requirements.txt
```

---

# Usage

## Image Detection

```bash
python inference/accident_detection.py --source path/to/image.jpg
```

## Video Detection

```bash
python inference/accident_detection.py --source path/to/video.mp4
```

## Custom Model Path

```bash
python inference/accident_detection.py --source path/to/video.mp4 --model path/to/model.pt
```

## Custom Confidence Threshold

```bash
python inference/accident_detection.py --source path/to/video.mp4 --conf 0.7
```

## Custom Output Path

```bash
python inference/accident_detection.py --source path/to/video.mp4 --output path/to/output.mp4
```

---

# Training

To train the YOLO26 model:

```bash
python training/detection/train_yolo26.py
```

This will:
- Load the dataset from `datasets/accident_detection/data.yaml`
- Train for 100 epochs with early stopping
- Save the best model to `results/detection/yolo26_baseline/weights/best.pt`

---

# Output

The system generates annotated output with:
- Bounding boxes around detected objects
- Class names displayed above each box
- Confidence scores for each detection

Output is automatically saved to `results/detection/output/` unless a custom path is specified.
