"""
Backend Inference Pipeline Integration.
Orchestrates low-light enhancement, accident detection, and severity estimation for backend API requests.
"""

from inference.accident_detection import AccidentDetector
from inference.enhancement import ImageEnhancer
from inference.severity_prediction import SeverityPredictor

class InferencePipeline:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.detector = AccidentDetector()
        self.severity_predictor = SeverityPredictor()

    def process_frame(self, frame):
        """
        Process a single image frame through the entire pipeline:
        1. Enhance frame (if required)
        2. Detect accident
        3. Identify accident type & severity
        """
        enhanced_frame = self.enhancer.enhance(frame)
        detection_result = self.detector.detect(enhanced_frame)
        
        if detection_result.get("accident_detected"):
            severity_result = self.severity_predictor.predict_severity(enhanced_frame)
            detection_result["severity_info"] = severity_result
            
        return detection_result

pipeline = InferencePipeline()
