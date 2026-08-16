"""
Zero-DCE + YOLO26 Integrated Accident Detection Pipeline with Severity Prediction.

Pipeline Overview:
    Low-light frame
        ↓
    Zero-DCE enhancement
        ↓
    Improved visual features
        ↓
    YOLO26 feature extraction
        ↓
    Accident classification
        ↓
    Temporal verification (5 consecutive frames)
        ↓
    Severity prediction (rule-based)
        ↓
    Display with severity information

This module integrates low-light enhancement with accident detection for night-time
traffic surveillance, and adds explainable severity estimation after accident confirmation.
"""

import sys
import os
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from inference.enhancement import ImageEnhancer
from inference.severity_prediction import SeverityPredictor


class ZeroDCEEnhancer:
    """
    Wrapper class for Zero-DCE low-light enhancement.
    Reuses the existing ImageEnhancer from inference/enhancement.py
    
    Adaptive Enhancement:
    Applies Zero-DCE only to low-light frames to prevent over-exposure
    of daylight videos. Uses brightness analysis to determine enhancement need.
    """
    
    # Brightness threshold for low-light detection (0-255 scale)
    LOW_LIGHT_THRESHOLD = 80
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize Zero-DCE enhancer.
        
        Args:
            model_path: Path to best_zero_dce.pth weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.base_dir = Path(__file__).resolve().parent.parent
        
        if model_path is None:
            model_path = (
                self.base_dir
                / "models"
                / "enhancement"
                / "zero_dce"
                / "best_zero_dce.pth"
            )
        
        model_path = Path(model_path)
        
        # Verify model path exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Zero-DCE model weights not found at: {model_path}\n"
                f"Please ensure the model is trained and saved at the correct location."
            )
        
        print(f"[+] Initializing Zero-DCE Enhancer...")
        print(f"    Model path: {model_path}")
        print(f"    Low-light threshold: {self.LOW_LIGHT_THRESHOLD}")
        
        self.enhancer = ImageEnhancer(model_path=model_path, device=device)
        
    def is_low_light(self, frame_bgr: np.ndarray) -> tuple:
        """
        Detect if frame is low-light based on average brightness.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Tuple of (is_low_light, brightness_value)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        brightness = gray.mean()
        
        return brightness < self.LOW_LIGHT_THRESHOLD, brightness
    
    def enhance_frame(self, frame_bgr: np.ndarray, frame_num: int = 0) -> tuple:
        """
        Adaptively enhance a frame using Zero-DCE based on illumination.
        
        Performance Optimization:
        Applies aspect-ratio preserving resize before Zero-DCE to improve FPS.
        Original resolution → 640x446 → Zero-DCE → YOLO26
        
        Args:
            frame_bgr: Input frame in BGR format (OpenCV default)
            frame_num: Frame number for debugging output
            
        Returns:
            Tuple of (enhanced_frame, original_frame, scale_factor)
        """
        # Store original frame for output
        original_frame = frame_bgr.copy()
        
        # Get original dimensions
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Check if frame is low-light
        is_low_light, brightness = self.is_low_light(frame_bgr)
        
        if is_low_light:
            # Performance optimization: Resize before Zero-DCE
            # Target max dimension of 640 while preserving aspect ratio
            max_dim = 640
            scale = min(max_dim / orig_w, max_dim / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Resize frame for enhancement
            resized_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Apply Zero-DCE enhancement for low-light frames
            enhanced_resized = self.enhancer.enhance_cv2(resized_frame)
            
            # Blend enhanced frame with original to prevent over-enhancement
            # 60% enhanced + 40% original preserves natural colors
            final_resized = cv2.addWeighted(enhanced_resized, 0.6, resized_frame, 0.4, 0)
            
            # Resize back to original dimensions for YOLO26
            enhanced_frame = cv2.resize(final_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            print(f"Frame {frame_num}: Brightness: {brightness:.1f} | Enhancement: APPLIED (low light) | Resize: {orig_w}x{orig_h} → {new_w}x{new_h}")
            return enhanced_frame, original_frame, scale
        else:
            # Skip enhancement for daylight frames
            print(f"Frame {frame_num}: Brightness: {brightness:.1f} | Enhancement: SKIPPED (daylight)")
            return frame_bgr, original_frame, 1.0


class AccidentDetector:
    """
    YOLO26-based accident detection model.
    Detects accidents and objects in enhanced frames.
    """
    
    # Accident classes that trigger alerts
    ACCIDENT_CLASSES = {
        'bike_bike_accident',
        'bike_object_accident',
        'bike_person_accident',
        'car_bike_accident',
        'car_car_accident',
        'car_object_accident',
        'car_person_accident'
    }
    
    def __init__(self, model_path: str = None, device: str = None, conf_threshold: float = 0.7):
        """
        Initialize YOLO26 accident detector.
        
        Args:
            model_path: Path to best.pt weights
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
        """
        self.base_dir = Path(__file__).resolve().parent.parent
        
        if model_path is None:
            model_path = (
                self.base_dir
                / "results"
                / "detection"
                / "yolo26_baseline"
                / "weights"
                / "best.pt"
            )
        
        model_path = Path(model_path)
        
        # Verify model path exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO26 model weights not found at: {model_path}\n"
                f"Please ensure the model is trained and saved at the correct location."
            )
        
        print(f"[+] Initializing YOLO26 Accident Detector...")
        print(f"    Model path: {model_path}")
        print(f"    Confidence threshold: {conf_threshold}")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load YOLO26 model
        self.model = YOLO(str(model_path))
        self.model.to(device)
        
        # Get class names from model
        self.class_names = self.model.names
        
        print(f"[+] Loaded YOLO26 model weights: {model_path}")
        print(f"    Device: {device}")
        print(f"    Classes: {len(self.class_names)}")
        
    def detect(self, frame: np.ndarray):
        """
        Run accident detection on a frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            YOLO results object
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        return results[0]
    
    def is_accident_detected(self, results) -> tuple:
        """
        Check if any accident class is detected.
        
        Args:
            results: YOLO results object
            
        Returns:
            Tuple of (is_accident, class_name, confidence)
        """
        if results is None or len(results.boxes) == 0:
            return False, None, 0.0
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])
            
            if class_name in self.ACCIDENT_CLASSES:
                return True, class_name, confidence
        
        return False, None, 0.0
    
    def get_all_detections(self, results) -> tuple:
        """
        Extract all detections from YOLO results for severity prediction.
        
        Args:
            results: YOLO results object
            
        Returns:
            Tuple of (object_count, object_classes_list)
        """
        if results is None or len(results.boxes) == 0:
            return 0, []
        
        object_count = len(results.boxes)
        object_classes = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            object_classes.append(class_name)
        
        return object_count, object_classes
    
    def draw_detections(self, frame: np.ndarray, results, severity_info: dict = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame with severity information.
        
        Args:
            frame: Input frame
            results: YOLO results object
            severity_info: Dictionary with severity prediction results (optional)
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        if results is None or len(results.boxes) == 0:
            return annotated_frame
        
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])
            
            # Choose color based on whether it's an accident
            if class_name in self.ACCIDENT_CLASSES:
                color = (0, 0, 255)  # Red for accidents
            else:
                color = (0, 255, 0)  # Green for normal objects
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Draw severity information if available
        if severity_info is not None:
            self._draw_severity_info(annotated_frame, severity_info)
        
        return annotated_frame
    
    def _draw_severity_info(self, frame: np.ndarray, severity_info: dict):
        """
        Draw severity information overlay on frame.
        
        Optimized version: Removes expensive transparency operations and reduces
        rendering overhead while maintaining all required information visibility.
        
        Args:
            frame: Input frame
            severity_info: Dictionary with severity prediction results
        """
        # Extract severity information
        accident_class = severity_info.get('accident_class', 'Unknown')
        severity = severity_info.get('severity', 'Unknown')
        priority = severity_info.get('priority', 'Unknown')
        
        # Choose color based on priority
        priority_colors = {
            'LOW': (0, 255, 0),        # Green
            'MEDIUM': (0, 165, 255),   # Orange
            'HIGH': (0, 0, 255),       # Red
            'IMMEDIATE': (0, 0, 128),  # Dark red
            'Unknown': (128, 128, 128) # Gray
        }
        color = priority_colors.get(priority, (128, 128, 128))
        
        # Draw solid background panel (optimized: no transparency, no frame copy)
        panel_height = 95  # Reduced from 120 (removed confidence line)
        panel_width = 350
        cv2.rectangle(
            frame,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),  # Solid black background
            -1
        )
        
        # Draw border
        cv2.rectangle(
            frame,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            color,
            2
        )
        
        # Draw text
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Accident Type
        cv2.putText(
            frame,
            f"ACCIDENT: {accident_class}",
            (20, y_offset),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        
        # Severity
        y_offset += 25
        cv2.putText(
            frame,
            f"SEVERITY: {severity}",
            (20, y_offset),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        
        # Priority
        y_offset += 25
        cv2.putText(
            frame,
            f"PRIORITY: {priority}",
            (20, y_offset),
            font,
            font_scale,
            color,
            font_thickness
        )


class VideoProcessor:
    """
    Main video processing pipeline with severity prediction.
    Coordinates Zero-DCE enhancement, YOLO26 detection, temporal verification, and severity estimation.
    
    Temporal Verification:
    Temporal verification reduces false positives by analysing multiple consecutive frames.
    Instead of alerting on a single frame detection, the system requires the accident
    class to be detected in ACCIDENT_CONFIRM_FRAMES consecutive frames before confirming.
    This filters out transient false detections while maintaining sensitivity to real accidents.
    
    Severity Prediction:
    Severity prediction is performed ONLY AFTER accident confirmation to avoid
    unnecessary processing and prevent severity assignment to false detections.
    """
    
    # Number of consecutive frames required to confirm an accident
    ACCIDENT_CONFIRM_FRAMES = 5
    
    def __init__(
        self,
        zero_dce_model_path: str = None,
        yolo26_model_path: str = None,
        device: str = None,
        conf_threshold: float = 0.7
    ):
        """
        Initialize the integrated pipeline with severity prediction.
        
        Args:
            zero_dce_model_path: Path to Zero-DCE weights
            yolo26_model_path: Path to YOLO26 weights
            device: Device to run inference on
            conf_threshold: Confidence threshold for YOLO26
        """
        print("=" * 70)
        print("  ZERO-DCE + YOLO26 + SEVERITY PREDICTION PIPELINE")
        print("=" * 70)
        
        # Initialize enhancer
        self.enhancer = ZeroDCEEnhancer(
            model_path=zero_dce_model_path,
            device=device
        )
        
        # Initialize detector
        self.detector = AccidentDetector(
            model_path=yolo26_model_path,
            device=device,
            conf_threshold=conf_threshold
        )
        
        # Initialize severity predictor
        print(f"[+] Initializing Severity Predictor...")
        self.severity_predictor = SeverityPredictor()
        print(f"[+] Severity Predictor initialized (rule-based, no model weights)")
        
        self.base_dir = Path(__file__).resolve().parent.parent
        
    def process_video(
        self,
        input_video_path: str,
        output_video_path: str = None,
        display_enhanced: bool = False
    ):
        """
        Process a video file through the integrated pipeline with severity prediction.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video (optional)
            display_enhanced: Whether to display enhanced frames during processing
        """
        input_path = Path(input_video_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
        # Set default output path based on input video name
        if output_video_path is None:
            output_dir = self.base_dir / "results" / "enhanced_detection_severity"
            output_dir.mkdir(parents=True, exist_ok=True)
            # Extract input video name and create output name
            input_filename = Path(input_video_path).stem
            output_video_path = output_dir / f"{input_filename}_severity_output.mp4"
        
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[+] Processing Video:")
        print(f"    Input:  {input_path}")
        print(f"    Output: {output_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"    FPS: {fps}")
        print(f"    Resolution: {width}x{height}")
        print(f"    Total frames: {total_frames}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing metrics
        frame_count = 0
        total_inference_time = 0.0
        
        # Profiling metrics
        zero_dce_times = []
        yolo26_times = []
        verification_times = []
        severity_times = []
        drawing_times = []
        video_write_times = []
        
        # Temporal verification metrics
        possible_accident_frames = 0  # Frames where accident class was detected
        confirmed_accidents = 0       # Accidents confirmed after consecutive detections
        consecutive_accident_count = 0  # Counter for consecutive accident frames
        
        # Severity tracking
        current_severity_info = None  # Store latest severity prediction
        current_accident_class = None  # Store current accident class
        current_confidence = 0.0       # Store current confidence
        
        print(f"\n[+] Starting frame processing...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Step 1: Adaptive Zero-DCE Enhancement
            # Brightness analysis → Conditional enhancement based on illumination
            # Performance optimization: Resize before Zero-DCE
            zero_dce_start = time.time()
            enhanced_frame, original_frame, scale = self.enhancer.enhance_frame(frame, frame_num=frame_count)
            zero_dce_times.append((time.time() - zero_dce_start) * 1000)
            
            # Step 2: YOLO26 Accident Detection
            # Enhanced frame → Feature extraction → Accident classification
            yolo26_start = time.time()
            results = self.detector.detect(enhanced_frame)
            yolo26_times.append((time.time() - yolo26_start) * 1000)
            
            # Extract all detections for severity prediction (needed after confirmation)
            verification_start = time.time()
            object_count, object_classes = self.detector.get_all_detections(results)
            
            # Check for accident detection with temporal verification
            is_accident, accident_class, confidence = self.detector.is_accident_detected(results)
            verification_times.append((time.time() - verification_start) * 1000)
            
            if is_accident:
                # Increment consecutive accident counter
                consecutive_accident_count += 1
                possible_accident_frames += 1
                
                # Store current accident info for severity prediction
                current_accident_class = accident_class
                current_confidence = confidence
                
                # Print possible accident detection
                print(f"Frame {frame_count}: Possible accident ({accident_class}, confidence: {confidence:.4f})")
                
                # Check if accident is confirmed after consecutive frames
                if consecutive_accident_count >= self.ACCIDENT_CONFIRM_FRAMES:
                    confirmed_accidents += 1
                    print(f"\n{'='*50}")
                    print(f"ACCIDENT CONFIRMED")
                    print(f"Class: {accident_class}")
                    print(f"Confidence: {confidence:.4f}")
                    print(f"Frame: {frame_count}")
                    print(f"{'='*50}")
                    
                    # Step 3: Severity Prediction (AFTER confirmation)
                    # Only predict severity for confirmed accidents
                    severity_start = time.time()
                    severity_result = self.severity_predictor.predict_severity(
                        class_name=accident_class,
                        confidence=confidence,
                        consecutive_frames=consecutive_accident_count,
                        detected_object_count=object_count,
                        detected_object_classes=object_classes
                    )
                    severity_times.append((time.time() - severity_start) * 1000)
                    
                    current_severity_info = severity_result
                    
                    print(f"\n{'='*50}")
                    print(f"SEVERITY PREDICTION")
                    print(f"Accident Type: {severity_result['accident_class']}")
                    print(f"Impact Level: {severity_result['impact_level']}")
                    print(f"Severity: {severity_result['severity']}")
                    print(f"Priority: {severity_result['priority']}")
                    print(f"Reason: {severity_result['reason']}")
                    print(f"{'='*50}\n")
            else:
                # Reset counter if no accident detected in current frame
                consecutive_accident_count = 0
            
            # Draw detections on enhanced frame with severity info
            drawing_start = time.time()
            annotated_frame = self.detector.draw_detections(
                enhanced_frame, 
                results, 
                severity_info=current_severity_info
            )
            drawing_times.append((time.time() - drawing_start) * 1000)
            
            # Write to output video
            video_write_start = time.time()
            out.write(annotated_frame)
            video_write_times.append((time.time() - video_write_start) * 1000)
            
            # Calculate inference time
            frame_time = time.time() - frame_start_time
            total_inference_time += frame_time
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                current_fps = frame_count / total_inference_time
                print(f"    Processed {frame_count}/{total_frames} frames | "
                      f"FPS: {current_fps:.2f} | Possible: {possible_accident_frames} | Confirmed: {confirmed_accidents}")
            
            # Optional display
            if display_enhanced:
                cv2.imshow('Enhanced Accident Detection with Severity', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
        avg_frame_time = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"  PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"[+] Total frames processed: {frame_count}")
        print(f"[+] Possible accident frames: {possible_accident_frames}")
        print(f"[+] Confirmed accidents: {confirmed_accidents}")
        print(f"[+] Average FPS: {avg_fps:.2f}")
        print(f"[+] Average inference time per frame: {avg_frame_time:.2f}ms")
        print(f"[+] Output saved to: {output_path}")

        # Build summary dict for callers (e.g. backend API)
        self._last_run_stats = {
            "frames_processed":     frame_count,
            "possible_accidents":   possible_accident_frames,
            "confirmed_accidents":  confirmed_accidents,
            "avg_fps":              round(avg_fps, 2),
            "avg_ms_per_frame":     round(avg_frame_time, 2),
            "last_severity":        current_severity_info,
        }
        
        # Profiling report
        print(f"\n{'='*70}")
        print(f"  PROFILING REPORT")
        print(f"{'='*70}")
        if zero_dce_times:
            print(f"[+] Zero-DCE Enhancement: {sum(zero_dce_times)/len(zero_dce_times):.2f}ms/frame")
        if yolo26_times:
            print(f"[+] YOLO26 Detection: {sum(yolo26_times)/len(yolo26_times):.2f}ms/frame")
        if verification_times:
            print(f"[+] Accident Verification: {sum(verification_times)/len(verification_times):.2f}ms/frame")
        if severity_times:
            print(f"[+] Severity Prediction: {sum(severity_times)/len(severity_times):.2f}ms/frame (called {len(severity_times)} times)")
        if drawing_times:
            print(f"[+] Drawing/Rendering: {sum(drawing_times)/len(drawing_times):.2f}ms/frame")
        if video_write_times:
            print(f"[+] Video Writing: {sum(video_write_times)/len(video_write_times):.2f}ms/frame")
        print(f"{'='*70}\n")
        
        return str(output_path)


def main():
    """
    Main entry point for the integrated pipeline with severity prediction.
    Accepts video filename via --video command-line argument.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Zero-DCE + YOLO26 + Severity Prediction Integrated Pipeline'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Video filename (e.g., test_video.mp4). Video should be in inference/videos/'
    )
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(
        conf_threshold=0.7
    )
    
    # Build input video path from argument
    base_dir = Path(__file__).resolve().parent.parent
    input_video = base_dir / "inference" / "videos" / args.video
    
    # Check if input video exists
    if not input_video.exists():
        print(f"[!] Input video not found at: {input_video}")
        
        # Show available .mp4 files in inference/videos/
        videos_dir = base_dir / "inference" / "videos"
        if videos_dir.exists():
            available_videos = list(videos_dir.glob("*.mp4"))
            if available_videos:
                print(f"[!] Available video files in {videos_dir}:")
                for video in available_videos:
                    print(f"    - {video.name}")
            else:
                print(f"[!] No .mp4 files found in {videos_dir}")
        else:
            print(f"[!] Videos directory not found: {videos_dir}")
        
        print(f"[!] Usage: python inference/enhanced_accident_detection_severity.py --video <video_filename.mp4>")
        return
    
    # Process video
    output_path = processor.process_video(
        input_video_path=str(input_video),
        display_enhanced=False  # Set to True to display frames during processing
    )
    
    print(f"[+] Pipeline execution completed successfully!")
    print(f"[+] Output video: {output_path}")


if __name__ == "__main__":
    main()
