"""
Diagnostic script to analyze Accident Detection model predictions on specific frames.
"""

import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
ACCIDENT_MODEL = ROOT_DIR / "results" / "detection" / "yolo26_baseline" / "weights" / "best.pt"
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "1.mp4"

# Accident classes to monitor
ACCIDENT_CLASSES = {
    'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
    'car_bike_accident', 'car_car_accident', 'car_object_accident', 'car_person_accident',
}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading model: {ACCIDENT_MODEL}")
    model = YOLO(str(ACCIDENT_MODEL))
    model.to(device)
    
    print(f"[*] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Total frames: {total_frames}")
    
    # Focus on frames 500-506 (around the confirmed accident at 502-505)
    start_frame = 500
    end_frame = 506
    
    print(f"\n[*] Analyzing frames {start_frame} to {end_frame}")
    print("=" * 80)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.5, device=device, verbose=False)[0]
        
        print(f"\nFrame {frame_num}:")
        accident_detections = []
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                
                if class_name in ACCIDENT_CLASSES:
                    accident_detections.append({
                        'class': class_name,
                        'conf': confidence,
                        'box': (x1, y1, x2, y2)
                    })
                    print(f"  {class_name}: conf={confidence:.3f}, box=({x1},{y1},{x2},{y2})")
        
        if not accident_detections:
            print("  No accident-class detections")
    
    cap.release()
    print("\n" + "=" * 80)
    print("[*] Diagnostic complete")

if __name__ == "__main__":
    main()
