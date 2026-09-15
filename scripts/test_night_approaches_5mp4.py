"""
Night Traffic detection improvement test for 5.mp4.
Tests 3 approaches: Original 640, Enhanced 640, Enhanced 1280.
"""

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from collections import Counter
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
NIGHT_MODEL = ROOT_DIR / "results" / "detection" / "night_traffic-2" / "weights" / "best.pt"
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "5.mp4"

TARGET_CLASSES = {'Bicycle', 'Bus', 'Car', 'Motorbike', 'People'}

def enhance_frame(frame):
    """Apply moderate CLAHE enhancement for night-time footage."""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def test_approach(approach_name, imgsz, use_enhancement, video_path, model, output_path):
    """Test a single approach."""
    print(f"\n[*] Testing: {approach_name}")
    print(f"    imgsz={imgsz}, enhancement={use_enhancement}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Statistics
    frames_with_detections = 0
    total_detections = 0
    class_counts = Counter()
    class_confidences = {cls: [] for cls in TARGET_CLASSES}
    
    # Output video
    output_dir = ROOT_DIR / "results" / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply enhancement if needed
        if use_enhancement:
            processed_frame = enhance_frame(frame)
        else:
            processed_frame = frame
        
        # Run inference
        results = model(processed_frame, conf=0.1, imgsz=imgsz, device='cuda', verbose=False)[0]
        
        frame_has_detections = False
        
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
                    class_confidences[class_name].append(confidence)
            
            # Draw detections on ORIGINAL frame (not enhanced)
            annotated = frame.copy()
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name in TARGET_CLASSES:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{class_name} {conf:.2f}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            annotated = frame
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    # Calculate statistics
    class_avg_conf = {}
    class_max_conf = {}
    for cls in TARGET_CLASSES:
        if class_confidences[cls]:
            class_avg_conf[cls] = sum(class_confidences[cls]) / len(class_confidences[cls])
            class_max_conf[cls] = max(class_confidences[cls])
        else:
            class_avg_conf[cls] = 0.0
            class_max_conf[cls] = 0.0
    
    # Ensure all target classes are in class_counts (even if zero)
    for cls in TARGET_CLASSES:
        if cls not in class_counts:
            class_counts[cls] = 0
    
    # Print results
    print(f"    Total frames: {total_frames}")
    print(f"    Frames with detections: {frames_with_detections}")
    print(f"    Total detections: {total_detections}")
    print(f"    Car: {class_counts['Car']} detections, avg conf: {class_avg_conf['Car']:.3f}, max conf: {class_max_conf['Car']:.3f}")
    print(f"    Motorbike: {class_counts['Motorbike']} detections, avg conf: {class_avg_conf['Motorbike']:.3f}, max conf: {class_max_conf['Motorbike']:.3f}")
    print(f"    Bicycle: {class_counts['Bicycle']} detections, avg conf: {class_avg_conf['Bicycle']:.3f}, max conf: {class_max_conf['Bicycle']:.3f}")
    print(f"    Bus: {class_counts['Bus']} detections, avg conf: {class_avg_conf['Bus']:.3f}, max conf: {class_max_conf['Bus']:.3f}")
    print(f"    People: {class_counts['People']} detections, avg conf: {class_avg_conf['People']:.3f}, max conf: {class_max_conf['People']:.3f}")
    print(f"    Output: {output_path}")
    
    return {
        'frames_with_detections': frames_with_detections,
        'total_detections': total_detections,
        'class_counts': dict(class_counts),
        'class_avg_conf': class_avg_conf,
        'class_max_conf': class_max_conf,
        'class_confidences': class_confidences
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading Night Traffic model: {NIGHT_MODEL}")
    model = YOLO(str(NIGHT_MODEL))
    model.to(device)
    
    print(f"[*] Video: {VIDEO_PATH}")
    
    output_dir = ROOT_DIR / "results" / "integration"
    
    # Test all three approaches
    results = {}
    
    # Approach 1: Original 640
    results['original_640'] = test_approach(
        "Original 640",
        640,
        False,
        VIDEO_PATH,
        model,
        output_dir / "5_night_original.mp4"
    )
    
    # Approach 2: Enhanced 640
    results['enhanced_640'] = test_approach(
        "Enhanced 640",
        640,
        True,
        VIDEO_PATH,
        model,
        output_dir / "5_night_enhanced.mp4"
    )
    
    # Approach 3: Enhanced 1280
    results['enhanced_1280'] = test_approach(
        "Enhanced 1280",
        1280,
        True,
        VIDEO_PATH,
        model,
        output_dir / "5_night_enhanced_1280.mp4"
    )
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Approach':<20} {'Frames w/ detections':<25} {'Cars':<10} {'Motorbikes':<12} {'People':<10} {'Best conf':<10}")
    print("-" * 80)
    
    for key in ['original_640', 'enhanced_640', 'enhanced_1280']:
        r = results[key]
        name = key.replace('_', ' ').title()
        frames = r['frames_with_detections']
        cars = r['class_counts']['Car']
        motorbikes = r['class_counts']['Motorbike']
        people = r['class_counts']['People']
        
        # Find best confidence across all classes
        best_conf = max(r['class_max_conf'].values())
        
        print(f"{name:<20} {frames:<25} {cars:<10} {motorbikes:<12} {people:<10} {best_conf:<10.3f}")
    
    print("=" * 80)
    
    # Answer questions
    print("\nANSWERS:")
    
    # 1. Which approach detects the most vehicles?
    vehicle_counts = {
        'original_640': results['original_640']['class_counts']['Car'] + results['original_640']['class_counts']['Motorbike'] + results['original_640']['class_counts']['Bicycle'] + results['original_640']['class_counts']['Bus'],
        'enhanced_640': results['enhanced_640']['class_counts']['Car'] + results['enhanced_640']['class_counts']['Motorbike'] + results['enhanced_640']['class_counts']['Bicycle'] + results['enhanced_640']['class_counts']['Bus'],
        'enhanced_1280': results['enhanced_1280']['class_counts']['Car'] + results['enhanced_1280']['class_counts']['Motorbike'] + results['enhanced_1280']['class_counts']['Bicycle'] + results['enhanced_1280']['class_counts']['Bus']
    }
    best_vehicles = max(vehicle_counts, key=vehicle_counts.get)
    print(f"1. Which approach detects the most vehicles? {best_vehicles} ({vehicle_counts[best_vehicles]} vehicles)")
    
    # 2. Which approach gives the best useful detections?
    # Best useful = most detections with reasonable confidence (>0.3)
    useful_counts = {}
    for key in ['original_640', 'enhanced_640', 'enhanced_1280']:
        r = results[key]
        useful = sum(1 for confs in r['class_confidences'].values() for c in confs if c > 0.3)
        useful_counts[key] = useful
    best_useful = max(useful_counts, key=useful_counts.get)
    print(f"2. Which approach gives the best useful detections? {best_useful} ({useful_counts[best_useful]} detections with conf > 0.3)")
    
    # 3. Does enhancement actually improve detection?
    if results['enhanced_640']['frames_with_detections'] > results['original_640']['frames_with_detections']:
        print("3. Does enhancement actually improve detection? YES")
    else:
        print("3. Does enhancement actually improve detection? NO")
    
    # 4. Does higher resolution improve detection?
    if results['enhanced_1280']['frames_with_detections'] > results['enhanced_640']['frames_with_detections']:
        print("4. Does higher resolution improve detection? YES")
    else:
        print("4. Does higher resolution improve detection? NO")
    
    # 5. Which approach should we use?
    print(f"5. Which approach should we use for the final Night Traffic inference? {best_vehicles}")

if __name__ == "__main__":
    main()
