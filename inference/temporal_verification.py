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


STATE_NO_INCIDENT = "NO_INCIDENT"
STATE_ACCIDENT_CANDIDATE = "ACCIDENT_CANDIDATE"
STATE_TEMPORAL_CONFIRMATION = "TEMPORAL_CONFIRMATION"
STATE_INCIDENT_ACTIVE_AFTERMATH = "INCIDENT_ACTIVE_AFTERMATH"
STATE_INCIDENT_CLOSED = "INCIDENT_CLOSED"


class TemporalVerifier:
    """
    Temporal verification for accident detection.
    
    Tracks accident detections across consecutive frames using an incident state machine:
    NO_INCIDENT -> ACCIDENT_CANDIDATE -> TEMPORAL_CONFIRMATION -> INCIDENT_ACTIVE_AFTERMATH -> INCIDENT_CLOSED.
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
    
    def __init__(self, confirm_frames: int = 3, gap_frames: int = 30):
        """
        Initialize temporal verifier.
        
        Args:
            confirm_frames: Number of consecutive frames required to confirm an accident
            gap_frames: Number of idle frames without detections to close an active incident
        """
        self.confirm_frames = confirm_frames
        self.gap_frames = gap_frames
        
        # Incident State Machine state
        self.state = STATE_NO_INCIDENT
        
        # Tracking state
        self.current_frame = 0
        self.consecutive_accident_count = 0
        self.aftermath_idle_count = 0
        self.current_accident_class = None
        self.current_accident_confidences = []
        
        # Statistics
        self.frames_with_accident_detections = 0
        self.possible_accident_events = 0
        self.confirmed_accidents: List[ConfirmedAccident] = []
        
        # Frame tracking
        self.current_event_start_frame = None
        self.last_detection_frame = None
    
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
            - is_confirmed: Whether an accident was confirmed IN THIS FRAME ONLY
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

        is_confirmed = False
        confirmed_class = None

        if self.state == STATE_NO_INCIDENT:
            if has_accident:
                self.state = STATE_ACCIDENT_CANDIDATE
                self.current_accident_class = frame_accident_class
                self.current_event_start_frame = self.current_frame
                self.last_detection_frame = self.current_frame
                self.consecutive_accident_count = 1
                self.current_accident_confidences = [frame_accident_confidence]
                self.aftermath_idle_count = 0
                self.possible_accident_events += 1
        
        elif self.state == STATE_ACCIDENT_CANDIDATE:
            if has_accident:
                if self.current_accident_class is None:
                    self.current_accident_class = frame_accident_class
                
                self.consecutive_accident_count += 1
                self.current_accident_confidences.append(frame_accident_confidence)
                self.last_detection_frame = self.current_frame

                if self.consecutive_accident_count >= self.confirm_frames:
                    # Accident confirmed! (Fires once per incident)
                    is_confirmed = True
                    confirmed_class = self.current_accident_class
                    self.state = STATE_INCIDENT_ACTIVE_AFTERMATH
                    self.aftermath_idle_count = 0

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
            else:
                # Candidate chain broken before confirmation
                self._reset_tracking()

        elif self.state == STATE_INCIDENT_ACTIVE_AFTERMATH:
            if has_accident:
                # Sustaining aftermath detection — reset idle counter
                self.aftermath_idle_count = 0
                self.last_detection_frame = self.current_frame
                self.consecutive_accident_count += 1
                self.current_accident_confidences.append(frame_accident_confidence)
                if self.confirmed_accidents:
                    self.confirmed_accidents[-1].duration_frames = self.consecutive_accident_count
            else:
                # Frame without accident detection — increment idle counter
                self.aftermath_idle_count += 1
                if self.aftermath_idle_count >= self.gap_frames:
                    # Incident closed after gap_frames of no detections
                    self.state = STATE_INCIDENT_CLOSED
                    self._reset_tracking()

        return has_accident, is_confirmed, confirmed_class
    
    def _reset_tracking(self):
        """Reset temporal tracking state."""
        self.consecutive_accident_count = 0
        self.aftermath_idle_count = 0
        self.current_accident_class = None
        self.current_accident_confidences = []
        self.current_event_start_frame = None
        self.last_detection_frame = None
        self.state = STATE_NO_INCIDENT
    
    def get_current_status(self) -> Tuple[str, int]:
        """
        Get current temporal status.
        
        Returns:
            (status_text, consecutive_count) tuple
        """
        if self.state == STATE_ACCIDENT_CANDIDATE:
            return f"POSSIBLE ACCIDENT ({self.consecutive_accident_count}/{self.confirm_frames})", self.consecutive_accident_count
        elif self.state == STATE_INCIDENT_ACTIVE_AFTERMATH:
            return f"INCIDENT ACTIVE / AFTERMATH", self.consecutive_accident_count
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
