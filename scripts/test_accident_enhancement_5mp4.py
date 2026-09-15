"""
Test whether CLAHE enhancement improves Accident Detection on 5.mp4.
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

def _enhance_frame(frame):
    """Apply moderate CLAHE enhancement for night-time footage."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def test_approach(approach_name, use_enhancement, confidence_threshold, video_path, model, output_path):
    """Test a single approach."""
    print(f"\n[*] Testing: {approach_name} (conf={confidence_threshold})")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Statistics
    frames_with_detections = 0
    total_detections = 0
    class_counts = Counter()
    class_confidences = {cls: [] for cls in ACCIDENT_CLASSES}
    best_confidence = 0.0
    best_class = None
    best_frame = None
    
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
            processed_frame = _enhance_frame(frame)
        else:
            processed_frame = frame
        
        # Run inference
        results = model(processed_frame, conf=confidence_threshold, imgsz=640, device='cuda', verbose=False)[0]
        
        frame_has_detections = False
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in ACCIDENT_CLASSES:
                    frame_has_detections = True
                    total_detections += 1
                    class_counts[class_name] += 1
                    class_confidences[class_name].append(confidence)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_class = class_name
                        best_frame = frame_num
            
            # Draw only accident-class boxes in RED on ORIGINAL frame
            annotated = frame.copy()
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
        
        if frame_has_detections:
            frames_with_detections += 1
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    # Print results
    print(f"    Total frames: {total_frames}")
    print(f"    Frames with accident detections: {frames_with_detections}")
    print(f"    Total accident detections: {total_detections}")
    
    if class_counts:
        print(f"    Accident classes detected:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            confs = class_confidences[class_name]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            max_conf = max(confs) if confs else 0.0
            print(f"      {class_name}: {count} detections, avg conf: {avg_conf:.3f}, max conf: {max_conf:.3f}")
    else:
        print(f"    Accident classes detected: NONE")
    
    print(f"    Best detection: {best_class} @ {best_confidence:.3f} (frame {best_frame})")
    print(f"    Output: {output_path}")
    
    return {
        'frames_with_detections': frames_with_detections,
        'total_detections': total_detections,
        'class_counts': dict(class_counts),
        'class_confidences': class_confidences,
        'best_confidence': best_confidence,
        'best_class': best_class,
        'best_frame': best_frame
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Loading Accident model: {ACCIDENT_MODEL}")
    model = YOLO(str(ACCIDENT_MODEL))
    model.to(device)
    
    print(f"[*] Video: {VIDEO_PATH}")
    
    output_dir = ROOT_DIR / "results" / "integration"
    
    confidence_thresholds = [0.5, 0.3, 0.1]
    
    # Test all combinations
    results = {}
    
    for conf in confidence_thresholds:
        # Original frame
        key_orig = f'original_conf{conf}'
        results[key_orig] = test_approach(
            f"Original frame (conf={conf})",
            False,
            conf,
            VIDEO_PATH,
            model,
            output_dir / f"5_accident_original_conf{conf}.mp4"
        )
        
        # Enhanced frame
        key_enh = f'enhanced_conf{conf}'
        results[key_enh] = test_approach(
            f"Enhanced frame (conf={conf})",
            True,
            conf,
            VIDEO_PATH,
            model,
            output_dir / f"5_accident_enhanced_conf{conf}.mp4"
        )
    
    # Final comparison at conf=0.5 (standard threshold)
    print("\n" + "=" * 80)
    print("FINAL COMPARISON (conf=0.5)")
    print("=" * 80)
    
    orig_05 = results['original_conf0.5']
    enh_05 = results['enhanced_conf0.5']
    
    print(f"\nOriginal frame:")
    print(f"  Accident detections: {orig_05['total_detections']}")
    print(f"  Best class: {orig_05['best_class']}")
    print(f"  Best confidence: {orig_05['best_confidence']:.3f}")
    
    print(f"\nEnhanced frame:")
    print(f"  Accident detections: {enh_05['total_detections']}")
    print(f"  Best class: {enh_05['best_class']}")
    print(f"  Best confidence: {enh_05['best_confidence']:.3f}")
    
    # Final conclusion
    print("\n" + "=" * 80)
    print("FINAL CONCLUSION")
    print("=" * 80)
    
    if enh_05['total_detections'] > 0 and orig_05['total_detections'] == 0:
        print("A. Enhancement makes Accident Detection work.")
    elif enh_05['total_detections'] > orig_05['total_detections']:
        print("B. Enhancement improves Accident Detection but detections remain weak.")
    else:
        print("C. Enhancement does not help; Accident Detection still gives zero detections.")

if __name__ == "__main__":
    main()
