"""
Inference runner wrapper for AI Smart Accident Detection System.

This module provides a wrapper around the existing inference pipeline
to integrate it with the FastAPI backend without modifying the core
inference code.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from inference.enhanced_accident_detection_severity import VideoProcessor


class InferenceRunner:
    """
    Wrapper class to run the existing accident detection pipeline
    from the FastAPI backend.
    """
    
    def __init__(self, device: str = None, conf_threshold: float = 0.7):
        """
        Initialize the inference runner.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for YOLO26
        """
        self.processor = VideoProcessor(
            device=device,
            conf_threshold=conf_threshold
        )
    
    def run_inference(
        self,
        input_video_path: str,
        output_video_path: str = None
    ) -> Dict[str, Any]:
        """
        Run accident detection inference on a video file.
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path to save output video (optional)
            
        Returns:
            Dictionary containing:
                - output_video_path: Path to generated output video
                - success: Boolean indicating if inference completed
                - error: Error message if inference failed
        """
        try:
            # Validate input video exists
            input_path = Path(input_video_path)
            if not input_path.exists():
                return {
                    "output_video_path": None,
                    "success": False,
                    "error": f"Input video not found: {input_video_path}"
                }
            
            # Run inference using existing pipeline
            output_path = self.processor.process_video(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                display_enhanced=False
            )

            # Collect per-run stats attached by process_video
            stats = getattr(self.processor, "_last_run_stats", {})

            return {
                "output_video_path": str(output_path),
                "success": True,
                "error": None,
                "stats": stats,
            }
            
        except Exception as e:
            return {
                "output_video_path": None,
                "success": False,
                "error": str(e)
            }
