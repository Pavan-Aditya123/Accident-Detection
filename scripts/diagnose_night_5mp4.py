"""
Night Object Model diagnostic for 5.mp4.
"""

import sys
from pathlib import Path
import cv2
import torch
from collections import Counter
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
NIGHT_MODEL = ROOT_DIR / "results" / "detection" / "night_traffic-2" / "weights" / "best.pt"
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "5.mp4"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading Night Object model: {NIGHT_MODEL}")
    model = YOLO(str(NIGHT_MODEL))
    model.to(device)
    
    print(f"[*] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[*] Total frames: {total_frames}")
    
    # Statistics
    frames_with_detections = 0
    class_counts = Counter()
    class_max_conf = {}
    
    # Output video
    output_dir = ROOT_DIR / "results" / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "5_night_diagnostic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\n[*] Processing frames...")
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.4, device=device, verbose=False)[0]
        
        frame_has_detections = False
        
        if len(results.boxes) > 0:
            frame_has_detections = True
            frames_with_detections += 1
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                class_counts[class_name] += 1
                
                if class_name not in class_max_conf or confidence > class_max_conf[class_name]:
                    class_max_conf[class_name] = confidence
            
            # Draw detections
            annotated = results.plot()
        else:
            annotated = frame
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    print("\n" + "=" * 80)
    print("NIGHT OBJECT MODEL DIAGNOSTIC RESULTS")
    print("=" * 80)
    print(f"\nTotal frames: {total_frames}")
    print(f"Frames with detections: {frames_with_detections}")
    
    print(f"\nDetections by class:")
    for class_name in ["Car", "Motorbike", "Bicycle", "Bus", "People"]:
        count = class_counts.get(class_name, 0)
        max_conf = class_max_conf.get(class_name, 0.0)
        print(f"  {class_name}: {count} detections, max confidence: {max_conf:.3f}")
    
    print(f"\nOutput video saved: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
