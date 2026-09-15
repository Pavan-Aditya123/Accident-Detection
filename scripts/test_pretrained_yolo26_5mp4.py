"""
Test pretrained YOLO26 on 5.mp4 to check object detection capability.
"""

import sys
from pathlib import Path
import cv2
import torch
from collections import Counter, defaultdict
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
YOLO26_MODEL = ROOT_DIR / "yolo26n.pt"
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "5.mp4"

# Classes to monitor
TARGET_CLASSES = {'car', 'motorcycle', 'bicycle', 'bus', 'person'}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading pretrained YOLO26: {YOLO26_MODEL}")
    model = YOLO(str(YOLO26_MODEL))
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
    total_detections = 0
    class_counts = Counter()
    class_max_conf = {}
    
    # Track frames with multiple relevant objects
    frames_with_car_motorcycle = []
    frames_with_car_person = []
    frames_with_motorcycle_person = []
    
    # Output video
    output_dir = ROOT_DIR / "results" / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "5_pretrained_yolo26.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\n[*] Processing frames...")
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.25, device=device, verbose=False)[0]
        
        frame_has_detections = False
        frame_classes = set()
        
        if len(results.boxes) > 0:
            frame_has_detections = True
            frames_with_detections += 1
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in TARGET_CLASSES:
                    total_detections += 1
                    class_counts[class_name] += 1
                    frame_classes.add(class_name)
                    
                    if class_name not in class_max_conf or confidence > class_max_conf[class_name]:
                        class_max_conf[class_name] = confidence
            
            # Check for multiple relevant objects
            if 'car' in frame_classes and 'motorcycle' in frame_classes:
                frames_with_car_motorcycle.append(frame_num)
            if 'car' in frame_classes and 'person' in frame_classes:
                frames_with_car_person.append(frame_num)
            if 'motorcycle' in frame_classes and 'person' in frame_classes:
                frames_with_motorcycle_person.append(frame_num)
            
            # Draw detections
            annotated = results.plot()
        else:
            annotated = frame
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    print("\n" + "=" * 80)
    print("PRETRAINED YOLO26 OBJECT DETECTION RESULTS")
    print("=" * 80)
    print(f"\n1. Total frames: {total_frames}")
    print(f"2. Frames with detections: {frames_with_detections}")
    print(f"3. Total detections: {total_detections}")
    
    print(f"\n4. Detection count for each class:")
    for class_name in sorted(TARGET_CLASSES):
        count = class_counts.get(class_name, 0)
        max_conf = class_max_conf.get(class_name, 0.0)
        print(f"   {class_name}: {count} detections, max confidence: {max_conf:.3f}")
    
    print(f"\n5. Maximum confidence for each class:")
    for class_name in sorted(TARGET_CLASSES):
        max_conf = class_max_conf.get(class_name, 0.0)
        print(f"   {class_name}: {max_conf:.3f}")
    
    print(f"\n6. Frames with multiple relevant objects:")
    print(f"   car + motorcycle: {len(frames_with_car_motorcycle)} frames")
    if frames_with_car_motorcycle:
        print(f"      Frames: {frames_with_car_motorcycle[:10]}")
        if len(frames_with_car_motorcycle) > 10:
            print(f"      ... and {len(frames_with_car_motorcycle) - 10} more")
    
    print(f"   car + person: {len(frames_with_car_person)} frames")
    if frames_with_car_person:
        print(f"      Frames: {frames_with_car_person[:10]}")
        if len(frames_with_car_person) > 10:
            print(f"      ... and {len(frames_with_car_person) - 10} more")
    
    print(f"   motorcycle + person: {len(frames_with_motorcycle_person)} frames")
    if frames_with_motorcycle_person:
        print(f"      Frames: {frames_with_motorcycle_person[:10]}")
        if len(frames_with_motorcycle_person) > 10:
            print(f"      ... and {len(frames_with_motorcycle_person) - 10} more")
    
    print(f"\nOutput video saved: {output_path}")
    print("=" * 80)
    
    # Final answer
    print("\nFINAL ANSWER:")
    if frames_with_detections > 0 and total_detections > 0:
        print("Can pretrained YOLO26 reliably see the vehicles/persons involved in the accident in 5.mp4?")
        print("YES")
        
        if frames_with_car_motorcycle or frames_with_car_person or frames_with_motorcycle_person:
            print("\nRelevant frames where vehicles are detected together:")
            if frames_with_car_motorcycle:
                print(f"  car + motorcycle: frames {frames_with_car_motorcycle[:5]}")
            if frames_with_car_person:
                print(f"  car + person: frames {frames_with_car_person[:5]}")
            if frames_with_motorcycle_person:
                print(f"  motorcycle + person: frames {frames_with_motorcycle_person[:5]}")
    else:
        print("Can pretrained YOLO26 reliably see the vehicles/persons involved in the accident in 5.mp4?")
        print("NO")

if __name__ == "__main__":
    main()
