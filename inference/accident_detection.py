"""
Basic YOLO26 Accident Detection Inference.

Simple pipeline: Image/Video -> YOLO26 -> Detection -> Output
"""

import argparse
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_verification import TemporalVerifier


def main():
    parser = argparse.ArgumentParser(description='Basic YOLO26 Accident Detection')
    parser.add_argument('--source', type=str, required=True, help='Path to image or video file')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLO26 model weights')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='Output path (auto-generated if not specified)')
    parser.add_argument('--temporal', action='store_true', help='Enable temporal verification')
    parser.add_argument('--confirm-frames', type=int, default=3, help='Consecutive frames required to confirm accident')
    args = parser.parse_args()

    # Set paths
    base_dir = Path(__file__).resolve().parent.parent
    
    if args.model is None:
        model_path = base_dir / "results" / "detection" / "yolo26_baseline" / "weights" / "best.pt"
    else:
        model_path = Path(args.model)

    source_path = Path(args.source)

    # Verify inputs
    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        sys.exit(1)

    if not source_path.exists():
        print(f"[!] Source not found: {source_path}")
        sys.exit(1)

    # Load model
    print(f"[+] Loading YOLO26 model from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[+] Device: {device}")
    
    model = YOLO(str(model_path))
    model.to(device)

    # Set output path
    if args.output is None:
        output_dir = base_dir / "results" / "detection" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{source_path.name}"
    else:
        output_path = Path(args.output)

    # Process based on source type
    source_ext = source_path.suffix.lower()
    
    if source_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Image processing
        print(f"[+] Processing image: {source_path}")
        results = model(str(source_path), conf=args.conf, device=device, verbose=False)
        
        # Draw detections
        result = results[0]
        annotated = result.plot()
        
        # Save result
        cv2.imwrite(str(output_path), annotated)
        print(f"[+] Output saved to: {output_path}")
        
        # Print detections
        if len(result.boxes) > 0:
            print(f"[+] Detections found: {len(result.boxes)}")
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"    Class: {class_name}, Confidence: {confidence:.4f}")
        else:
            print("[+] No detections found")
    
    elif source_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video processing
        print(f"[+] Processing video: {source_path}")
        cap = cv2.VideoCapture(str(source_path))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"    FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}, Duration: {duration:.2f}s")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        frames_with_accidents = 0
        accident_classes_detected = set()
        total_inference_time = 0.0
        
        import time
        
        # Initialize temporal verifier if enabled
        verifier = None
        if args.temporal:
            verifier = TemporalVerifier(confirm_frames=args.confirm_frames)
            print(f"[+] Temporal verification enabled (confirm_frames={args.confirm_frames})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            results = model(frame, conf=args.conf, device=device, verbose=False)
            inference_time = time.time() - frame_start
            total_inference_time += inference_time
            
            result = results[0]
            annotated = result.plot()
            
            # Extract detections for temporal verification
            frame_detections = []
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    frame_detections.append((class_name, confidence))
            
            # Process temporal verification if enabled
            if verifier:
                has_accident, is_confirmed, confirmed_class = verifier.process_frame(frame_detections)
                
                # Add temporal status to frame
                status_text, consecutive_count = verifier.get_current_status()
                if status_text:
                    cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if is_confirmed:
                    cv2.putText(annotated, "ACCIDENT CONFIRMED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            out.write(annotated)
            
            if len(result.boxes) > 0:
                total_detections += len(result.boxes)
                
                # Check for accident detections (baseline stats)
                accident_classes = {
                    'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
                    'car_bike_accident', 'car_car_accident', 'car_object_accident', 'car_person_accident'
                }
                
                frame_has_accident = False
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name in accident_classes:
                        accident_classes_detected.add(class_name)
                        frame_has_accident = True
                
                if frame_has_accident:
                    frames_with_accidents += 1
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"    Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        avg_inference_time = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
        processing_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
        
        print(f"[+] Output saved to: {output_path}")
        print(f"[+] Video Statistics:")
        print(f"    Total frames: {total_frames}")
        print(f"    Video FPS: {fps}")
        print(f"    Video duration: {duration:.2f}s")
        print(f"    Frames processed: {frame_count}")
        print(f"    Average inference time: {avg_inference_time:.2f}ms/frame")
        print(f"    Processing FPS: {processing_fps:.2f}")
        print(f"    Frames with accident detections: {frames_with_accidents}")
        print(f"    Total detections: {total_detections}")
        print(f"    Accident classes detected: {sorted(accident_classes_detected)}")
        
        # Print temporal verification report if enabled
        if verifier:
            verifier.print_report()
            
            # Save temporal results to JSON
            temporal_output = base_dir / "results" / "detection" / "output" / f"temporal_{source_path.stem}.json"
            verifier.save_results(str(temporal_output))
    
    else:
        print(f"[!] Unsupported file format: {source_ext}")
        sys.exit(1)

    print("[+] Processing complete")


if __name__ == "__main__":
    main()
