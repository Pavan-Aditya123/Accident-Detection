"""
Accident Severity Prediction Module.

Rule-based, explainable severity estimation derived directly from
YOLO26 detection outputs. No trained ML model required.

Inputs (all available from the existing detection pipeline):
    - accident_class_name   : str   — YOLO26 class name
    - confidence_score      : float — YOLO26 detection confidence (0–1)
    - consecutive_frames    : int   — frames the accident has been continuously detected
    - detected_object_count : int   — total objects detected in the same frame
    - detected_object_classes: list[str] — class names of all objects in the frame

Output:
    dict with keys: accident_class, impact_level, severity, priority,
                    confidence, reason
"""

from __future__ import annotations
from typing import Optional


# ── Impact weight table ────────────────────────────────────────────────────────
# Derived from YOLO26 class analysis (Step 4A).
# bike_object_accident has near-zero recall (AP50=0.045) — capped separately.
IMPACT_WEIGHTS: dict[str, int] = {
    "bike_object_accident":  1,   # Low    — bike vs inanimate object
    "car_object_accident":   1,   # Low    — car  vs inanimate object
    "bike_bike_accident":    3,   # Medium — two-wheeler collision
    "car_car_accident":      3,   # Medium — vehicle-to-vehicle
    "car_bike_accident":     4,   # High   — asymmetric mass impact
    "bike_person_accident":  5,   # Very High — human involvement
    "car_person_accident":   5,   # Very High — pedestrian strike
}

IMPACT_LABELS: dict[int, str] = {
    1: "Low",
    3: "Medium",
    4: "High",
    5: "Very High",
}

# ── Severity scale (ordered list for index arithmetic) ────────────────────────
SEVERITY_LEVELS = ["Minor", "Moderate", "Major", "Critical"]

# ── Base severity from impact weight ──────────────────────────────────────────
BASE_SEVERITY: dict[int, str] = {
    1: "Minor",
    3: "Moderate",
    4: "Major",
    5: "Critical",
}

# ── Priority mapping ──────────────────────────────────────────────────────────
PRIORITY_MAP: dict[str, str] = {
    "Minor":    "LOW",
    "Moderate": "MEDIUM",
    "Major":    "HIGH",
    "Critical": "IMMEDIATE",
}

# ── Safety constants ──────────────────────────────────────────────────────────
# Classes that must never be downgraded below Major (human involvement)
HUMAN_INVOLVEMENT_CLASSES = {"car_person_accident", "bike_person_accident"}

# Classes whose maximum severity is capped at Minor (unreliable detection)
UNRELIABLE_CLASSES = {"bike_object_accident"}

# Confidence thresholds for modifiers
CONF_HIGH_THRESHOLD = 0.85
CONF_LOW_THRESHOLD  = 0.72

# Duration threshold for sustained detection
DURATION_THRESHOLD  = 15   # consecutive frames

# Co-detection threshold
OBJECT_COUNT_THRESHOLD = 2  # objects in same frame


def _clamp_severity(level: str) -> str:
    """Ensure severity stays within the defined scale."""
    if level not in SEVERITY_LEVELS:
        return "Minor"
    return level


def _shift_severity(current: str, delta: int) -> str:
    """Shift severity up (+1) or down (−1), clamped to scale boundaries."""
    idx = SEVERITY_LEVELS.index(current)
    idx = max(0, min(len(SEVERITY_LEVELS) - 1, idx + delta))
    return SEVERITY_LEVELS[idx]


def predict_severity(
    class_name: str,
    confidence: float,
    consecutive_frames: int,
    detected_object_count: int = 0,
    detected_object_classes: Optional[list] = None,
) -> dict:
    """
    Predict accident severity using a deterministic rule system.

    Parameters
    ----------
    class_name : str
        YOLO26 accident class name (e.g. 'car_car_accident').
    confidence : float
        Detection confidence score in [0, 1].
    consecutive_frames : int
        Number of consecutive frames the accident class has been detected.
    detected_object_count : int
        Total number of YOLO26 boxes detected in the same frame (including
        the accident box itself).
    detected_object_classes : list[str], optional
        Class names of all detections in the frame. Not used in current
        rules but preserved for future extension.

    Returns
    -------
    dict
        Keys: accident_class, impact_level, severity, priority,
              confidence, reason
    """
    if detected_object_classes is None:
        detected_object_classes = []

    reasons: list[str] = []

    # ── 1. Resolve impact weight ───────────────────────────────────────────────
    weight = IMPACT_WEIGHTS.get(class_name, None)

    if weight is None:
        # Unknown class — treat conservatively as Medium
        weight = 3
        impact_level = "Medium"
        reasons.append(f"Unknown accident class '{class_name}' treated as Medium impact")
    else:
        impact_level = IMPACT_LABELS[weight]

    # ── 2. Base severity from weight ──────────────────────────────────────────
    severity = BASE_SEVERITY[weight]
    reasons.append(f"Base severity '{severity}' from impact weight {weight} ({impact_level})")

    # ── 3. Confidence modifier ────────────────────────────────────────────────
    if confidence >= CONF_HIGH_THRESHOLD:
        severity = _shift_severity(severity, +1)
        reasons.append(
            f"Confidence {confidence:.3f} >= {CONF_HIGH_THRESHOLD} → severity increased"
        )
    elif confidence < CONF_LOW_THRESHOLD:
        severity = _shift_severity(severity, -1)
        reasons.append(
            f"Confidence {confidence:.3f} < {CONF_LOW_THRESHOLD} → severity decreased"
        )

    # ── 4. Duration modifier ──────────────────────────────────────────────────
    if consecutive_frames >= DURATION_THRESHOLD:
        severity = _shift_severity(severity, +1)
        reasons.append(
            f"Sustained detection ({consecutive_frames} consecutive frames "
            f">= {DURATION_THRESHOLD}) → severity increased"
        )

    # ── 5. Object involvement modifier ────────────────────────────────────────
    if detected_object_count >= OBJECT_COUNT_THRESHOLD:
        severity = _shift_severity(severity, +1)
        reasons.append(
            f"{detected_object_count} objects detected in frame "
            f"(>= {OBJECT_COUNT_THRESHOLD}) → severity increased"
        )

    # ── 6. Safety rules (applied after all modifiers) ─────────────────────────

    # Rule 1: human-involvement floor — never below Major
    if class_name in HUMAN_INVOLVEMENT_CLASSES:
        idx = SEVERITY_LEVELS.index(severity)
        floor_idx = SEVERITY_LEVELS.index("Major")
        if idx < floor_idx:
            severity = "Major"
            reasons.append(
                f"Human involvement ({class_name}) → severity floor enforced (minimum Major)"
            )

    # Rule 2: unreliable class cap — never above Minor
    if class_name in UNRELIABLE_CLASSES:
        severity = "Minor"
        reasons.append(
            f"'{class_name}' has near-zero model recall → severity capped at Minor"
        )

    # ── 7. Final clamp (defensive) ────────────────────────────────────────────
    severity = _clamp_severity(severity)

    # ── 8. Priority ───────────────────────────────────────────────────────────
    priority = PRIORITY_MAP[severity]

    return {
        "accident_class": class_name,
        "impact_level":   impact_level,
        "severity":       severity,
        "priority":       priority,
        "confidence":     round(confidence, 4),
        "reason":         ". ".join(reasons),
    }


# ── Backwards-compatible SeverityPredictor class ──────────────────────────────
class SeverityPredictor:
    """
    Class wrapper around predict_severity() for compatibility with
    backend/inference.py which imports this class by name.
    """

    def __init__(self, model_path: str = None):
        # No model weights needed — rule-based system
        self.model_path = model_path

    def predict_severity(
        self,
        class_name: str,
        confidence: float,
        consecutive_frames: int,
        detected_object_count: int = 0,
        detected_object_classes: Optional[list] = None,
    ) -> dict:
        return predict_severity(
            class_name=class_name,
            confidence=confidence,
            consecutive_frames=consecutive_frames,
            detected_object_count=detected_object_count,
            detected_object_classes=detected_object_classes,
        )


if __name__ == "__main__":
    predictor = SeverityPredictor()
    print("Severity Predictor Module Initialized (rule-based, no model weights required).")
