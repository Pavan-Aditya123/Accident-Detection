# AI Smart Accident Detection System

An AI-based accident detection framework that combines Zero-DCE low-light enhancement with YOLO26 accident detection and rule-based severity estimation to improve night-time road accident analysis.

---

# Features

- YOLO26 based accident detection
- Zero-DCE based night-time image enhancement
- Low-light accident detection improvement
- Aspect-ratio preserving resolution optimization
- Temporal accident confirmation using consecutive frames
- Accident severity estimation
- Emergency priority classification
- Annotated output video generation

---

# System Architecture

```
Input Video
↓
Low-Light Detection
↓
Zero-DCE Enhancement
↓
YOLO26 Accident Detection
↓
5-Frame Temporal Verification
↓
Severity Prediction
↓
Emergency Priority Output
↓
Processed Video
```

---

# Project Workflow

1. Video frames are processed sequentially.
2. Dark frames are enhanced using Zero-DCE.
3. Enhanced frames are passed to YOLO26.
4. Accident classes are detected.
5. Temporal verification reduces false alarms.
6. Severity is calculated only after confirmed accident detection.
7. Final output video displays accident type, severity, and priority.

---

# Technologies Used

**Deep Learning:**
- YOLO26
- Zero-DCE

**Programming:**
- Python

**Libraries:**
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

**Hardware:**
- NVIDIA GPU with CUDA support

---

# YOLO26 Accident Classes

| Class |
|---|
| bike_bike_accident |
| bike_object_accident |
| bike_person_accident |
| car_bike_accident |
| car_car_accident |
| car_object_accident |
| car_person_accident |

---

# Night-Time Enhancement

Zero-DCE improves illumination in low-light scenes while preserving details instead of simply increasing brightness.

- Natural enhancement
- Low-light visibility improvement
- Integration before YOLO26 detection

---

# Severity Prediction Module

Severity is estimated using an explainable rule-based system.

**Inputs:**
- Accident class
- Confidence score
- Consecutive detection frames
- Number of detected objects

**Severity levels:**

| Severity | Emergency Priority |
|---|---|
| Minor | LOW |
| Moderate | MEDIUM |
| Major | HIGH |
| Critical | IMMEDIATE |

---

# Performance Results

## Night Accident Detection Comparison

| Method | Accident Detections |
|---|---:|
| YOLO26 Only | 49 |
| Zero-DCE + YOLO26 | 230 |

## Optimization Results

| Metric | Result |
|---|---:|
| Zero-DCE input resolution | 1786×1246 → 640×446 |
| FPS improvement | 3.91 → 16.48 |
| GPU memory reduction | 85.9% |
| Zero-DCE inference reduction | 83.9% |

## Final Integrated Pipeline

- Severity module integrated successfully
- Optimized severity visualization implemented
- Final severity pipeline achieved approximately 10.74 FPS during integrated testing
- Detection and severity outputs remained unchanged

---

# Project Structure

```
AI-Smart-Accident-Detection-System/

├── inference/
│   ├── enhanced_accident_detection.py
│   ├── enhanced_accident_detection_severity.py
│   ├── severity_prediction.py
│   ├── enhancement.py
│   └── accident_detection.py
│
├── training/
│
├── models/
│
├── datasets/
│
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone <repository-url>

cd AI-Smart-Accident-Detection-System

pip install -r requirements.txt
```
