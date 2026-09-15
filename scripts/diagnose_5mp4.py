"""
Comprehensive diagnostic for Accident Detection model on 5.mp4.
"""

import sys
from pathlib import Path
import cv2
import torch
from collections import defaultdict, Counter
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
ACCIDENT_MODEL = ROOT_DIR / "results" / "detection" / "yolo26_baseline" / "weights" / "best.pt"
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "5.mp4"

# Accident classes
ACCIDENT_CLASSES = {
    'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
    'car_bike_accident', 'car_car_accident', 'car_object_accident', 'car_person_accident',
}

# Normal classes
NORMAL_CLASSES = {'car', 'bike', 'person'}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading model: {ACCIDENT_MODEL}")
    model = YOLO(str(ACCIDENT_MODEL))
    model.to(device)
    
    print(f"[*] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Total frames: {total_frames}")
    
    # Statistics
    frames_with_any_detection = 0
    frames_with_accident_detection = 0
    frames_with_normal_detection = 0
    
    all_detections = Counter()  # class name -> count
    accident_class_max_conf = defaultdict(float)  # accident class -> max confidence
    accident_class_max_frame = defaultdict(int)  # accident class -> frame number
    
    # Track detections below 0.5 confidence
    low_conf_accident_detections = []  # (frame, class, confidence)
    
    # Track normal objects in frames
    normal_objects_per_frame = []
    
    print(f"\n[*] Processing all frames...")
    print("=" * 80)
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.5, device=device, verbose=False)[0]
        
        frame_has_any = False
        frame_has_accident = False
        frame_has_normal = False
        frame_normal_classes = set()
        
        if len(results.boxes) > 0:
            frame_has_any = True
            frames_with_any_detection += 1
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                all_detections[class_name] += 1
                
                if class_name in ACCIDENT_CLASSES:
                    frame_has_accident = True
                    if confidence > accident_class_max_conf[class_name]:
                        accident_class_max_conf[class_name] = confidence
                        accident_class_max_frame[class_name] = frame_num
                elif class_name in NORMAL_CLASSES:
                    frame_has_normal = True
                    frame_normal_classes.add(class_name)
        
        if frame_has_accident:
            frames_with_accident_detection += 1
        
        if frame_has_normal:
            frames_with_normal_detection += 1
            normal_objects_per_frame.append((frame_num, frame_normal_classes))
    
    cap.release()
    
    # Now check for low-confidence accident detections (below 0.5)
    print(f"[*] Checking for accident detections below 0.5 confidence...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.1, device=device, verbose=False)[0]  # Lower threshold for diagnostic
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in ACCIDENT_CLASSES and confidence < 0.5:
                    low_conf_accident_detections.append((frame_num, class_name, confidence))
    
    cap.release()
    
    # Report
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)
    
    print(f"\n1. Total frames: {total_frames}")
    print(f"2. Frames with ANY detection (including normal classes): {frames_with_any_detection}")
    print(f"3. Frames with accident-class detection: {frames_with_accident_detection}")
    print(f"4. Frames with normal-class detection (car/bike/person): {frames_with_normal_detection}")
    
    print(f"\n5. ALL detected classes and their counts (confidence >= 0.5):")
    for class_name, count in sorted(all_detections.items(), key=lambda x: -x[1]):
        print(f"   {class_name}: {count} detections")
    
    print(f"\n6. Maximum confidence for each accident class:")
    for class_name in sorted(ACCIDENT_CLASSES):
        if class_name in accident_class_max_conf:
            print(f"   {class_name}: {accident_class_max_conf[class_name]:.3f} (frame {accident_class_max_frame[class_name]})")
        else:
            print(f"   {class_name}: NO DETECTION")
    
    if accident_class_max_conf:
        best_class = max(accident_class_max_conf, key=accident_class_max_conf.get)
        print(f"\n7. Frame with highest accident-class confidence:")
        print(f"   Frame {accident_class_max_frame[best_class]}: {best_class} @ {accident_class_max_conf[best_class]:.3f}")
    else:
        print(f"\n7. Frame with highest accident-class confidence: N/A (no accident detections)")
    
    print(f"\n8. Accident-class detections below confidence 0.5:")
    if low_conf_accident_detections:
        print(f"   Found {len(low_conf_accident_detections)} low-confidence accident detections:")
        for frame_num, class_name, conf in low_conf_accident_detections[:20]:  # Show first 20
            print(f"   Frame {frame_num}: {class_name} @ {conf:.3f}")
        if len(low_conf_accident_detections) > 20:
            print(f"   ... and {len(low_conf_accident_detections) - 20} more")
    else:
        print(f"   NONE")
    
    print(f"\n9. Normal objects detected (car/bike/person):")
    if frames_with_normal_detection > 0:
        print(f"   Frames with normal objects: {frames_with_normal_detection}")
        print(f"   Sample frames with normal objects:")
        for frame_num, classes in normal_objects_per_frame[:10]:
            print(f"   Frame {frame_num}: {', '.join(sorted(classes))}")
        if len(normal_objects_per_frame) > 10:
            print(f"   ... and {len(normal_objects_per_frame) - 10} more frames")
    else:
        print(f"   NONE")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    # Conclusion
    print("\nCONCLUSION:")
    if frames_with_accident_detection > 0:
        print("A. Model detects accident with confidence >= 0.5: YES")
    elif low_conf_accident_detections:
        print("A. Model detects accident but confidence below 0.5: YES")
    else:
        print("A. Model detects accident but confidence below 0.5: NO")
    
    if frames_with_normal_detection > 0:
        print("B. Model detects normal objects but fails to classify as accident: YES")
    else:
        print("B. Model detects normal objects but fails to classify as accident: NO")
    
    if frames_with_accident_detection > 0:
        print("C. Model detects wrong accident class: POSSIBLE (check class names above)")
    else:
        print("C. Model detects wrong accident class: NO (no accident detections)")
    
    if frames_with_accident_detection == 0 and not low_conf_accident_detections:
        print("D. Model completely misses the accident: YES")
    else:
        print("D. Model completely misses the accident: NO")

if __name__ == "__main__":
    main()
