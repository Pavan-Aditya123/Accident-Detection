"""
Accident Model diagnostic for 5.mp4.
"""

import sys
from pathlib import Path
import cv2
import torch
from collections import Counter
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
ACCIDENT_MODEL = ROOT_DIR / "results" / "detection" / "yolo26_baseline" / "weights" / "best.pt"
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "5.mp4"

ACCIDENT_CLASSES = {
    'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
    'car_bike_accident', 'car_car_accident', 'car_object_accident', 'car_person_accident',
}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading Accident model: {ACCIDENT_MODEL}")
    model = YOLO(str(ACCIDENT_MODEL))
    model.to(device)
    
    print(f"[*] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[*] Total frames: {total_frames}")
    
    # Statistics
    frames_with_accident = 0
    accident_class_counts = Counter()
    accident_class_max_conf = {}
    accident_class_max_frame = {}
    
    # Output video
    output_dir = ROOT_DIR / "results" / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "5_accident_diagnostic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\n[*] Processing frames...")
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.5, device=device, verbose=False)[0]
        
        frame_has_accident = False
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in ACCIDENT_CLASSES:
                    frame_has_accident = True
                    accident_class_counts[class_name] += 1
                    
                    if class_name not in accident_class_max_conf or confidence > accident_class_max_conf[class_name]:
                        accident_class_max_conf[class_name] = confidence
                        accident_class_max_frame[class_name] = frame_num
            
            # Draw only accident-class boxes in red
            annotated = frame.copy()
            if len(results.boxes) > 0:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name in ACCIDENT_CLASSES:
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, f"{class_name} {conf:.2f}", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            annotated = frame
        
        if frame_has_accident:
            frames_with_accident += 1
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    print("\n" + "=" * 80)
    print("ACCIDENT MODEL DIAGNOSTIC RESULTS")
    print("=" * 80)
    print(f"\nTotal frames: {total_frames}")
    print(f"Frames with accident-class detection: {frames_with_accident}")
    
    print(f"\nAccident classes detected:")
    for class_name in sorted(ACCIDENT_CLASSES):
        count = accident_class_counts.get(class_name, 0)
        max_conf = accident_class_max_conf.get(class_name, 0.0)
        max_frame = accident_class_max_frame.get(class_name, "N/A")
        print(f"  {class_name}: {count} detections, max confidence: {max_conf:.3f} (frame {max_frame})")
    
    print(f"\nOutput video saved: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
