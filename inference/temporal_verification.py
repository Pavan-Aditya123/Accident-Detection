"""
Temporal Accident Verification Module.

Converts frame-level YOLO26n detections into video-level accident events
using consecutive frame consistency checking.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ConfirmedAccident:
    """Data class for a confirmed accident event."""
    accident_class: str
    first_detection_frame: int
    confirmation_frame: int
    duration_frames: int
    avg_confidence: float
    max_confidence: float
    confirmation_latency: int


class TemporalVerifier:
    """
    Temporal verification for accident detection.
    
    Tracks accident detections across consecutive frames and confirms
    accidents only when they persist for a configurable number of frames.
    """
    
    # Accident classes that trigger temporal verification
    ACCIDENT_CLASSES = {
        'bike_bike_accident',
        'bike_object_accident',
        'bike_person_accident',
        'car_bike_accident',
        'car_car_accident',
        'car_object_accident',
        'car_person_accident'
    }
    
    def __init__(self, confirm_frames: int = 5):
        """
        Initialize temporal verifier.
        
        Args:
            confirm_frames: Number of consecutive frames required to confirm an accident
        """
        self.confirm_frames = confirm_frames
        
        # Tracking state
        self.current_frame = 0
        self.consecutive_accident_count = 0
        self.current_accident_class = None
        self.current_accident_confidences = []
        
        # Statistics
        self.frames_with_accident_detections = 0
        self.possible_accident_events = 0
        self.confirmed_accidents: List[ConfirmedAccident] = []
        
        # For tracking first detection of current possible event
        self.current_event_start_frame = None
    
    def process_frame(
        self,
        detections: List[Tuple[str, float]]
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Process a single frame's detections.
        
        Args:
            detections: List of (class_name, confidence) tuples for this frame
            
        Returns:
            (has_accident, is_confirmed, confirmed_class) tuple
            - has_accident: Whether this frame has an accident detection
            - is_confirmed: Whether an accident was confirmed in this frame
            - confirmed_class: The class of the confirmed accident (if confirmed)
        """
        self.current_frame += 1
        
        # Check if any accident class is detected in this frame
        frame_accident_class = None
        frame_accident_confidence = 0.0
        
        for class_name, confidence in detections:
            if class_name in self.ACCIDENT_CLASSES:
                frame_accident_class = class_name
                frame_accident_confidence = max(frame_accident_confidence, confidence)
        
        has_accident = frame_accident_class is not None
        
        if has_accident:
            self.frames_with_accident_detections += 1
        
        # Temporal verification logic
        if has_accident:
            # Check if this is the same accident class as before
            if self.current_accident_class is None:
                # Starting a new possible event
                self.current_accident_class = frame_accident_class
                self.current_event_start_frame = self.current_frame
                self.possible_accident_events += 1
            elif self.current_accident_class != frame_accident_class:
                # Different accident class - reset and start new event
                self._reset_tracking()
                self.current_accident_class = frame_accident_class
                self.current_event_start_frame = self.current_frame
                self.possible_accident_events += 1
            
            # Increment counter and track confidence
            self.consecutive_accident_count += 1
            self.current_accident_confidences.append(frame_accident_confidence)
            
            # Check if accident is confirmed
            is_confirmed = False
            confirmed_class = None
            
            if self.consecutive_accident_count >= self.confirm_frames:
                # Accident confirmed!
                is_confirmed = True
                confirmed_class = self.current_accident_class
                
                # Calculate statistics
                avg_conf = sum(self.current_accident_confidences) / len(self.current_accident_confidences)
                max_conf = max(self.current_accident_confidences)
                duration = self.consecutive_accident_count
                latency = self.current_frame - self.current_event_start_frame
                
                confirmed_accident = ConfirmedAccident(
                    accident_class=self.current_accident_class,
                    first_detection_frame=self.current_event_start_frame,
                    confirmation_frame=self.current_frame,
                    duration_frames=duration,
                    avg_confidence=avg_conf,
                    max_confidence=max_conf,
                    confirmation_latency=latency
                )
                
                self.confirmed_accidents.append(confirmed_accident)
                
                # Reset after confirmation (to detect new events)
                self._reset_tracking()
            
            return has_accident, is_confirmed, confirmed_class
        else:
            # No accident detected - reset tracking
            self._reset_tracking()
            return has_accident, False, None
    
    def _reset_tracking(self):
        """Reset temporal tracking state."""
        self.consecutive_accident_count = 0
        self.current_accident_class = None
        self.current_accident_confidences = []
        self.current_event_start_frame = None
    
    def get_current_status(self) -> Tuple[str, int]:
        """
        Get current temporal status.
        
        Returns:
            (status_text, consecutive_count) tuple
            - status_text: "POSSIBLE ACCIDENT" or ""
            - consecutive_count: Current consecutive frame count
        """
        if self.consecutive_accident_count > 0:
            return f"POSSIBLE ACCIDENT ({self.consecutive_accident_count}/{self.confirm_frames})", self.consecutive_accident_count
        return "", 0
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            'total_frames': self.current_frame,
            'frames_with_accident_detections': self.frames_with_accident_detections,
            'possible_accident_events': self.possible_accident_events,
            'confirmed_accident_events': len(self.confirmed_accidents),
            'confirmed_accidents': [asdict(acc) for acc in self.confirmed_accidents]
        }
    
    def print_report(self):
        """Print a detailed report of temporal verification results."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("  TEMPORAL VERIFICATION REPORT")
        print("=" * 70)
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with accident detections: {stats['frames_with_accident_detections']}")
        print(f"Number of possible accident events: {stats['possible_accident_events']}")
        print(f"Number of confirmed accident events: {stats['confirmed_accident_events']}")
        
        if stats['confirmed_accident_events'] > 0:
            print(f"\nConfirmed Accident Events:")
            for i, acc in enumerate(stats['confirmed_accidents'], 1):
                print(f"\n  Event {i}:")
                print(f"    Accident class: {acc['accident_class']}")
                print(f"    Start frame: {acc['first_detection_frame']}")
                print(f"    Confirmation frame: {acc['confirmation_frame']}")
                print(f"    Duration: {acc['duration_frames']} frames")
                print(f"    Average confidence: {acc['avg_confidence']:.4f}")
                print(f"    Maximum confidence: {acc['max_confidence']:.4f}")
                print(f"    Confirmation latency: {acc['confirmation_latency']} frames")
        
        print("=" * 70 + "\n")
    
    def save_results(self, output_path: str):
        """
        Save temporal verification results to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        stats = self.get_statistics()
        
        # Add metadata
        stats['metadata'] = {
            'confirm_frames': self.confirm_frames,
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[+] Temporal verification results saved to: {output_file}")
