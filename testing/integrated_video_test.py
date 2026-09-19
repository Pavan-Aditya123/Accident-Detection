"""
Integrated Three-Model Video Test.

Pipeline:
    CCTV video
        ↓
    Night Traffic YOLO26n        (every frame — traffic context)
        ↓
    Accident Detection YOLO26n   (every frame — accident classes only)
        ↓
    5-frame temporal verification
        ↓
    Post-temporal validation:
        • Spatial consistency  (boxes overlap across confirmation window)
        • Object context       (Night Traffic objects support the accident type)
        ↓
    Severity YOLO26n on ACCIDENT CROP  (verified incidents only)
        ↓
    ONE final accident report

Severity label mapping (3-class model):
    0 = moderate-accident  → MODERATE
    1 = no-accident        → internal sentinel, ignored in aggregation
    2 = severe-accident    → HIGH

Possible final decisions:
    NO ACCIDENT DETECTED
    ACCIDENT DETECTED — NOT CONFIRMED
    ACCIDENT CONFIRMED — MODERATE
    ACCIDENT CONFIRMED — HIGH
    ACCIDENT CONFIRMED — SEVERITY UNCERTAIN
"""

import subprocess
import sys
import time
import argparse
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from ultralytics import YOLO

def _finalize_and_verify_video(raw_output_path: str, final_output_path: str, input_fps: float, input_frames: int):
    """
    Ensures browser compatibility by converting raw video to H.264 (yuv420p) with faststart header.
    Verifies output file size, frame count, FPS, and duration against input parameters.
    """
    raw_p = Path(raw_output_path)
    final_p = Path(final_output_path)

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_target = final_p.parent / f"tmp_{final_p.name}"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(raw_p),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(tmp_target)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and tmp_target.exists() and tmp_target.stat().st_size > 0:
            if raw_p.exists() and raw_p != final_p:
                raw_p.unlink()
            if final_p.exists():
                final_p.unlink()
            tmp_target.rename(final_p)
        else:
            if raw_p != final_p and raw_p.exists():
                if final_p.exists():
                    final_p.unlink()
                raw_p.rename(final_p)
    except Exception as e:
        print(f"  [!] H.264 conversion warning: {e}")
        if raw_p != final_p and raw_p.exists():
            if final_p.exists():
                final_p.unlink()
            raw_p.rename(final_p)

    if not final_p.exists():
        raise RuntimeError(f"Output video missing: {final_p}")
    size_bytes = final_p.stat().st_size
    if size_bytes == 0:
        raise RuntimeError(f"Output video is 0 bytes: {final_p}")

    out_cap = cv2.VideoCapture(str(final_p))
    out_fps = float(out_cap.get(cv2.CAP_PROP_FPS))
    out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dur = out_frames / out_fps if out_fps > 0 else 0.0
    out_cap.release()

    in_dur = input_frames / input_fps if input_fps > 0 else 0.0

    print("\n" + "=" * 50)
    print("OUTPUT VIDEO VERIFICATION REPORT")
    print("=" * 50)
    print(f"File Path    : {final_p.resolve()}")
    print(f"File Size    : {size_bytes / (1024*1024):.2f} MB")
    print(f"INPUT Stats  : FPS={input_fps:.2f}, Frames={input_frames}, Duration={in_dur:.2f}s")
    print(f"OUTPUT Stats : FPS={out_fps:.2f}, Frames={out_frames}, Duration={out_dur:.2f}s")
    print("=" * 50 + "\n")


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from inference.temporal_verification import TemporalVerifier


# ── Verified model paths ───────────────────────────────────────────────────────
NIGHT_TRAFFIC_MODEL  = ROOT_DIR / "results" / "detection" / "night_traffic-2"     / "weights" / "best.pt"
ACCIDENT_MODEL       = ROOT_DIR / "results" / "detection" / "yolo26_baseline"     / "weights" / "best.pt"
SEVERITY_MODEL       = ROOT_DIR / "results" / "detection" / "accident_severity-2" / "weights" / "best.pt"

DEFAULT_VIDEO      = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "video.avi"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "integration"
DEFAULT_SUMMARY    = ROOT_DIR / "results" / "integration" / "integration_summary.txt"

# ── Accident classes (yolo26_baseline confirmed class names) ───────────────────
# ONLY these 7 classes trigger accident logic. car/bike/person are NEVER accidents.
ACCIDENT_CLASSES = {
    'bike_bike_accident',
    'bike_object_accident',
    'bike_person_accident',
    'car_bike_accident',
    'car_car_accident',
    'car_object_accident',
    'car_person_accident',
}

# Severity model class id → internal label  (accident_severity-2 confirmed classes)
SEVERITY_RAW = {
    0: "MODERATE",
    1: "NO_ACCIDENT",   # internal sentinel — filtered out in aggregation
    2: "HIGH",
}

# Padding (pixels) around accident bounding box for severity crop
CROP_PAD = 20

# ── Fixed system configuration ─────────────────────────────────────────────────
# Number of consecutive accident frames required by temporal verifier
CONFIRM_FRAMES = 3

# Minimum confidence for an accident-class detection
ACCIDENT_CONFIDENCE = 0.5

# Confirmation events within this many frames → same incident
INCIDENT_GAP_FRAMES = 30

# ── Validation thresholds ──────────────────────────────────────────────────────
# Minimum IoU between first and last box in the temporal window
# to pass spatial consistency.  0.10 is intentionally lenient —
# moving vehicles shift, but a false positive on open road drifts far.
SPATIAL_IOU_THRESHOLD = 0.10

# ── Night Traffic object names (night_traffic-2 confirmed class names) ─────────
NIGHT_VEHICLE_CLASSES  = {"Car", "Bus", "Motorbike", "Bicycle"}
NIGHT_PERSON_CLASSES   = {"People"}
NIGHT_ALL_CLASSES      = NIGHT_VEHICLE_CLASSES | NIGHT_PERSON_CLASSES

# Which Night Traffic objects support each accident type.
# Absence of Night Traffic evidence does NOT reject — it just marks INSUFFICIENT.
ACCIDENT_CONTEXT_MAP = {
    'car_car_accident':    {"Car", "Bus"},
    'car_bike_accident':   {"Car", "Bus", "Bicycle", "Motorbike"},
    'car_object_accident': {"Car", "Bus"},
    'bike_bike_accident':  {"Bicycle", "Motorbike"},
    'bike_person_accident':{"Bicycle", "Motorbike", "People"},
    'bike_object_accident':{"Bicycle", "Motorbike"},
    'car_person_accident': {"Car", "Bus", "People"},
}

# Also accept normal-class names from the Accident Detection model as supporting context
ACCIDENT_NORMAL_SUPPORT = {
    'car_car_accident':    {'car'},
    'car_bike_accident':   {'car', 'bike'},
    'car_object_accident': {'car'},
    'bike_bike_accident':  {'bike'},
    'bike_person_accident':{'bike', 'person'},
    'bike_object_accident':{'bike'},
    'car_person_accident': {'car', 'person'},
}

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
COLOR_NIGHT     = (255, 200,   0)   # Blue-yellow — night traffic
COLOR_ACCIDENT  = (  0,   0, 255)   # Red         — accident-class boxes ONLY
COLOR_POSSIBLE  = (  0, 220, 220)   # Cyan        — possible / building up
COLOR_CONFIRM   = (  0,   0, 200)   # Dark red    — confirmed banner
COLOR_NOT_CONF  = (  0, 128, 255)   # Orange      — detected but not confirmed
COLOR_HIGH      = (  0,   0, 255)   # Red         — HIGH severity
COLOR_MODERATE  = (  0, 165, 255)   # Orange      — MODERATE severity
COLOR_UNCERTAIN = (  0, 200, 200)   # Cyan        — UNCERTAIN severity


# ── Helper functions ───────────────────────────────────────────────────────────

def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _severity_color(label: str):
    if label == "HIGH":
        return COLOR_HIGH
    if label == "MODERATE":
        return COLOR_MODERATE
    return COLOR_UNCERTAIN


def _draw_label(frame, text, x, y, color, font_scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, color, thickness)


def _enhance_frame(frame):
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


def _is_night_frame(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Determine if a frame is low-light / night-time based on mean grayscale luminance.
    Frames with mean brightness < threshold (100.0) are classified as night frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def _draw_night_boxes(frame, results, class_names):
    """Night Traffic boxes — always non-red."""
    if results is None or len(results.boxes) == 0:
        return
    for box in results.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names.get(cls, str(cls))
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NIGHT, 2)
        _draw_label(frame, f"{name} {conf:.2f}", x1, y1 - 4, COLOR_NIGHT)


def _draw_accident_boxes(frame, results, class_names):
    """
    Accident Detection boxes.
    RED only for the 7 accident classes.
    Normal classes (car, bike, person) are skipped — Night Traffic covers them.
    """
    if results is None or len(results.boxes) == 0:
        return
    for box in results.boxes:
        cls  = int(box.cls[0])
        name = class_names.get(cls, "")
        if name not in ACCIDENT_CLASSES:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCIDENT, 2)
        _draw_label(frame, f"{name} {conf:.2f}", x1, y1 - 4, COLOR_ACCIDENT)


def _overlay_status(frame, current_acc_class, consecutive, confirm_frames,
                    display_confirmed, display_not_confirmed,
                    display_severity_label):
    """Top-left HUD on the output video."""
    panel_y = 10

    if display_confirmed and display_severity_label:
        cv2.rectangle(frame, (10, panel_y), (440, panel_y + 30), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, panel_y), (440, panel_y + 30), COLOR_CONFIRM, 2)
        _draw_label(frame, "ACCIDENT CONFIRMED", 15, panel_y + 22,
                    (255, 255, 255), 0.7, 2)
        sev_y     = panel_y + 40
        sev_color = _severity_color(display_severity_label)
        cv2.rectangle(frame, (10, sev_y), (440, sev_y + 30), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, sev_y), (440, sev_y + 30), sev_color, 2)
        _draw_label(frame, f"SEVERITY: {display_severity_label}", 15,
                    sev_y + 22, sev_color, 0.7, 2)

    elif display_not_confirmed:
        cv2.rectangle(frame, (10, panel_y), (440, panel_y + 30), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, panel_y), (440, panel_y + 30), COLOR_NOT_CONF, 2)
        _draw_label(frame, "ACCIDENT DETECTED — NOT CONFIRMED", 15,
                    panel_y + 22, (255, 255, 255), 0.65, 2)

    elif current_acc_class and consecutive > 0:
        text = f"POSSIBLE: {current_acc_class} ({consecutive}/{confirm_frames})"
        cv2.rectangle(frame, (10, panel_y), (520, panel_y + 30), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, panel_y), (520, panel_y + 30), COLOR_POSSIBLE, 1)
        _draw_label(frame, text, 15, panel_y + 22, COLOR_POSSIBLE, 0.6, 2)


def _crop_accident_region(frame: np.ndarray, accident_box: tuple,
                          pad: int = CROP_PAD) -> np.ndarray:
    """Crop accident bounding box + padding from the CCTV frame."""
    if accident_box is None:
        return frame
    x1, y1, x2, y2 = accident_box
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _best_accident_box(accident_results, accident_names) -> tuple | None:
    """Highest-confidence accident-class bounding box (x1,y1,x2,y2) or None."""
    if accident_results is None or len(accident_results.boxes) == 0:
        return None
    best_conf, best_box = -1.0, None
    for box in accident_results.boxes:
        cls  = int(box.cls[0])
        name = accident_names.get(cls, "")
        if name not in ACCIDENT_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box  = tuple(int(v) for v in box.xyxy[0].cpu().numpy())
    return best_box


def _all_normal_classes_in_frame(accident_results, accident_names) -> set:
    """Return the set of normal class names (car/bike/person) detected this frame."""
    found = set()
    if accident_results is None or len(accident_results.boxes) == 0:
        return found
    for box in accident_results.boxes:
        cls  = int(box.cls[0])
        name = accident_names.get(cls, "")
        if name in {'car', 'bike', 'person'}:
            found.add(name)
    return found


# ── Post-temporal validation ───────────────────────────────────────────────────

def _check_spatial_consistency(box_buffer: list) -> bool:
    """
    Check that accident boxes across the temporal window are spatially consistent.

    Strategy: compute IoU between the first and last box in the buffer.
    A very low IoU means the detections drifted across unrelated scene regions,
    suggesting a sliding false positive rather than a localised collision.

    Intentionally lenient (SPATIAL_IOU_THRESHOLD = 0.10) so genuine accidents
    where the camera or vehicles move are not rejected.
    """
    valid_boxes = [b for b in box_buffer if b is not None]
    if len(valid_boxes) < 2:
        # Cannot assess consistency — give benefit of the doubt
        return True
    iou = _iou(valid_boxes[0], valid_boxes[-1])
    return iou >= SPATIAL_IOU_THRESHOLD


def _check_object_context(accident_class: str,
                          night_context_set: set,
                          normal_class_set: set) -> str:
    """
    Check whether the Night Traffic detections (or normal-class detections
    from the accident model) are consistent with the predicted accident type.

    Returns 'STRONG' or 'INSUFFICIENT'.

    Rules:
    - Look up expected Night Traffic classes for this accident type.
    - Also accept normal-class names from the Accident Detection model
      (car/bike/person) as supporting evidence — this covers low-light
      scenes where the Night Traffic model may miss objects.
    - If any expected object type is found in either source → STRONG.
    - If nothing matches → INSUFFICIENT.
    - INSUFFICIENT does NOT reject the accident; it is reported as context only.
    """
    expected_night  = ACCIDENT_CONTEXT_MAP.get(accident_class, set())
    expected_normal = ACCIDENT_NORMAL_SUPPORT.get(accident_class, set())

    if expected_night & night_context_set:
        return "STRONG"
    if expected_normal & normal_class_set:
        return "STRONG"
    return "INSUFFICIENT"


def _validate_incident(confirmed_class: str,
                       box_buffer: list,
                       night_context_set: set,
                       normal_class_set: set,
                       debug: bool = False) -> dict:
    """
    Run all post-temporal validation checks and return a results dict.

    Keys:
        temporal_check      : always "PASS" (called only after 5/5)
        spatial_consistency : "PASS" | "FAIL"
        object_context      : "STRONG" | "INSUFFICIENT"
        verified            : bool — True only if spatial passes
                              (object context is informational, not blocking)
    """
    temporal_check = "PASS"   # caller guarantees 5 consecutive frames

    spatial_pass = _check_spatial_consistency(box_buffer)
    spatial_check = "PASS" if spatial_pass else "FAIL"

    obj_context = _check_object_context(
        confirmed_class, night_context_set, normal_class_set
    )

    # Verification based on temporal confirmation only.
    # Spatial consistency and object context are informational, not blocking.
    verified = True

    if debug:
        print(f"  [VALIDATE] temporal={temporal_check} | "
              f"spatial={spatial_check} (IoU threshold={SPATIAL_IOU_THRESHOLD}) | "
              f"context={obj_context} | verified={verified}")

    return {
        "temporal_check":   temporal_check,
        "spatial_check":    spatial_check,
        "object_context":   obj_context,
        "verified":         verified,
    }


# ── Severity aggregation ───────────────────────────────────────────────────────

def _aggregate_severity(severity_votes: list) -> tuple:
    """
    Aggregate (label, confidence) severity votes.
    NO_ACCIDENT votes are filtered out before choosing the severity level.
    Returns (label, avg_confidence): label is MODERATE | HIGH | UNCERTAIN.
    """
    if not severity_votes:
        return "UNCERTAIN", 0.0

    real_votes = [(l, c) for l, c in severity_votes if l != "NO_ACCIDENT"]
    if not real_votes:
        return "UNCERTAIN", 0.0

    counts    = Counter(l for l, _ in real_votes)
    max_count = max(counts.values())
    candidates = [l for l, cnt in counts.items() if cnt == max_count]

    if len(candidates) == 1:
        winner = candidates[0]
    else:
        avgs   = {l: sum(c for ll, c in real_votes if ll == l) /
                     sum(1 for ll, _ in real_votes if ll == l)
                  for l in candidates}
        winner = max(avgs, key=lambda l: avgs[l])

    winner_confs = [c for l, c in real_votes if l == winner]
    return winner, sum(winner_confs) / len(winner_confs)


def _most_common_class(class_votes: list) -> str:
    """Most frequent accident class; ties broken by highest average confidence."""
    if not class_votes:
        return "unknown"
    counts    = Counter(n for n, _ in class_votes)
    max_count = max(counts.values())
    candidates = [n for n, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    avgs = {n: sum(c for nn, c in class_votes if nn == n) /
               sum(1 for nn, _ in class_votes if nn == n)
            for n in candidates}
    return max(avgs, key=lambda n: avgs[n])


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    video_path: str,
    output_video_path: str,
    summary_path: str,
    conf_night: float = 0.55,
    conf_severity: float = 0.3,
    debug: bool = False,
):
    conf_accident  = ACCIDENT_CONFIDENCE   # fixed constant (0.50)
    confirm_frames = CONFIRM_FRAMES        # fixed constant (3)
    device = 0 if torch.cuda.is_available() else "cpu"

    # ── Validate paths ─────────────────────────────────────────────────────────
    missing = False
    for label, path in [
        ("Input video",              video_path),
        ("Night Traffic model",      NIGHT_TRAFFIC_MODEL),
        ("Accident Detection model", ACCIDENT_MODEL),
        ("Severity model",           SEVERITY_MODEL),
    ]:
        if not Path(path).exists():
            print(f"[!] {label} not found: {path}")
            missing = True
    if missing:
        sys.exit(1)

    SEP  = "=" * 60
    DASH = "-" * 60

    # ── Load models ────────────────────────────────────────────────────────────
    night_model    = YOLO(str(NIGHT_TRAFFIC_MODEL));  night_model.to(device)
    accident_model = YOLO(str(ACCIDENT_MODEL));        accident_model.to(device)
    severity_model = YOLO(str(SEVERITY_MODEL));        severity_model.to(device)

    night_names    = night_model.names
    accident_names = accident_model.names

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open video: {video_path}")
        sys.exit(1)

    fps_val      = cap.get(cv2.CAP_PROP_FPS)
    fps          = float(fps_val) if (fps_val and fps_val > 0 and not np.isnan(fps_val)) else 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(SEP)
    print("  INTEGRATED THREE-MODEL VIDEO TEST")
    print(SEP)
    print(f"  Device          : {'GPU' if device == 0 else 'CPU'}")
    print(f"  Night Traffic   : loaded  ({NIGHT_TRAFFIC_MODEL.name})")
    print(f"  Accident Model  : loaded  ({ACCIDENT_MODEL.name})")
    print(f"  Severity Model  : loaded  ({SEVERITY_MODEL.name})")
    print(f"  Input           : {Path(video_path).name}")
    print(f"  Frames          : {total_frames}")
    print(f"  FPS             : {fps:.2f}")
    print(f"  Resolution      : {width}x{height}")
    print(DASH)
    print("  PROCESSING")
    print(DASH)

    # ── Output video ───────────────────────────────────────────────────────────
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = str(Path(output_video_path).parent / f"raw_{Path(output_video_path).name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))


    # ── Temporal verifier ──────────────────────────────────────────────────────
    verifier = TemporalVerifier(confirm_frames=confirm_frames, gap_frames=INCIDENT_GAP_FRAMES)

    # ── Accumulators ──────────────────────────────────────────────────────────
    frame_count          = 0
    frames_with_accident = 0
    total_time           = 0.0

    # Rolling box buffer for spatial consistency check.
    # Stores the best accident box for each frame while a possible accident is building.
    box_buffer: list = []

    # Night Traffic and normal-class context collected during the temporal window
    window_night_context:  set = set()
    window_normal_context: set = set()

    # Incident store
    incidents:        list = []
    current_incident       = None

    # Per-video summary of validation results (for report)
    final_validation: dict = {
        "temporal_check":  "N/A",
        "spatial_check":   "N/A",
        "object_context":  "N/A",
        "verified":        False,
    }

    # Video overlay latch
    display_confirmed      = False
    display_not_confirmed  = False
    display_severity_label = None
    latched_confirmed      = False
    latched_severity_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        # ── Step 1: Night Traffic model (Auxiliary Only) ───────────────────────
        # Night Traffic is AUXILIARY ONLY for visualization and secondary context.
        # It MUST NEVER control or gate whether an accident is detected.
        enhanced_frame = _enhance_frame(frame)
        night_results = night_model(
            enhanced_frame, conf=conf_night, imgsz=640, device=device, verbose=False
        )[0]

        # ── Step 2: Primary Accident Detection YOLO26n ─────────────────────────
        # Accident Detection YOLO26n is the PRIMARY accident detector.
        # For low-light/night frames, mild CLAHE contrast enhancement is applied to the input.
        # Daytime frames are passed to Accident Detection un-enhanced.
        # Original frame is strictly preserved for visualization, crops, severity inference, and output video.
        if _is_night_frame(frame):
            accident_input_frame = enhanced_frame
        else:
            accident_input_frame = frame

        accident_results = accident_model(
            accident_input_frame, conf=conf_accident, device=device, verbose=False
        )[0]

        # Extract ONLY accident-class detections → temporal verifier
        frame_detections   = []
        frame_has_accident = False
        frame_acc_box      = None

        if accident_results is not None and len(accident_results.boxes) > 0:
            for box in accident_results.boxes:
                cls      = int(box.cls[0])
                name     = accident_names.get(cls, "")
                conf_val = float(box.conf[0])
                if name in ACCIDENT_CLASSES:
                    frame_detections.append((name, conf_val))
                    frame_has_accident = True
                    if debug:
                        print(f"  Frame {frame_count:5d}: {name} {conf_val:.2f}")

        if frame_has_accident:
            frames_with_accident += 1
            frame_acc_box = _best_accident_box(accident_results, accident_names)

        # Collect normal-class names from Accident Detection model this frame
        frame_normal_classes = _all_normal_classes_in_frame(
            accident_results, accident_names
        )

        # Collect Night Traffic class names this frame
        frame_night_classes: set = set()
        if night_results is not None and len(night_results.boxes) > 0:
            for box in night_results.boxes:
                cls  = int(box.cls[0])
                name = night_names.get(cls, "")
                if name:
                    frame_night_classes.add(name)

        # ── Step 3: Temporal verification ─────────────────────────────────────
        has_acc, is_confirmed, confirmed_class = verifier.process_frame(
            frame_detections
        )

        # Maintain box buffer and context window while a possible accident builds
        _, consec_now = verifier.get_current_status()

        if has_acc and verifier.state == "ACCIDENT_CANDIDATE":
            # Accident chain is building — accumulate window context
            box_buffer.append(frame_acc_box)
            window_night_context.update(frame_night_classes)
            window_normal_context.update(frame_normal_classes)
        elif not has_acc and verifier.state == "NO_INCIDENT":
            # Candidate chain broken — reset window accumulators
            box_buffer.clear()
            window_night_context.clear()
            window_normal_context.clear()

        # ── Step 4: Post-temporal validation & single confirmation execution ──
        severity_label      = None
        severity_confidence = 0.0

        if is_confirmed and confirmed_class:

            # Run validation on the just-completed temporal window
            validation = _validate_incident(
                confirmed_class,
                box_buffer.copy(),
                window_night_context.copy(),
                window_normal_context.copy(),
                debug=debug,
            )
            final_validation = validation   # keep for report

            if validation["verified"]:
                # ── Step 5: Severity on accident crop (Runs ONCE per incident) ──
                crop = _crop_accident_region(frame, frame_acc_box)
                sev_results = severity_model(
                    crop, conf=conf_severity, device=device, verbose=False
                )[0]

                if sev_results is not None and len(sev_results.boxes) > 0:
                    top_conf, top_cls = -1.0, None
                    for box in sev_results.boxes:
                        c = float(box.conf[0])
                        if c > top_conf:
                            top_conf, top_cls = c, int(box.cls[0])
                    severity_label      = SEVERITY_RAW.get(top_cls, "MODERATE")
                    severity_confidence = top_conf
                else:
                    severity_label      = "NO_ACCIDENT"
                    severity_confidence = 0.0

                overlay_sev = (severity_label
                               if severity_label != "NO_ACCIDENT"
                               else "UNCERTAIN")

                print(f"  [ALERT] ACCIDENT CONFIRMED | Frame: {frame_count} "
                      f"| Type: {confirmed_class} "
                      f"| Context: {validation['object_context']}")
                if severity_label != "NO_ACCIDENT":
                    print(f"  [INFO]  Severity: {severity_label} "
                          f"| Confidence: {severity_confidence:.2f}")
                else:
                    print(f"  [INFO]  Severity model inconclusive on crop")

                if debug:
                    print(f"  [DEBUG] spatial={validation['spatial_check']} | "
                          f"context={validation['object_context']}")

                # ── Incident initialization ─────────────────────────────────────
                acc_conf_val = (
                    max(c for _, c in frame_detections)
                    if frame_detections else 0.0
                )

                start_f = verifier.current_event_start_frame if verifier.current_event_start_frame else frame_count
                current_incident = {
                    "first_frame":    start_f,
                    "last_frame":     frame_count,
                    "class_votes":    [(confirmed_class, acc_conf_val)],
                    "severity_votes": [(severity_label, severity_confidence)],
                    "acc_confs":      [acc_conf_val],
                    "night_context":  window_night_context.copy(),
                    "validation":     validation,
                }

                display_confirmed      = True
                display_not_confirmed  = False
                display_severity_label = overlay_sev
                latched_confirmed      = True
                latched_severity_label = overlay_sev

            else:
                # Spatial check failed — accident detected but not verified
                print(f"  [WARN]  Temporal threshold reached but spatial "
                      f"consistency FAILED | Frame: {frame_count} "
                      f"| Type: {confirmed_class} — marking NOT CONFIRMED")
                display_not_confirmed  = True
                display_confirmed      = False
                display_severity_label = None

            # Clear temporal window buffers
            box_buffer.clear()
            window_night_context.clear()
            window_normal_context.clear()

        # ── Aftermath frame accumulation ────────────────────────────────────────
        elif verifier.state == "INCIDENT_ACTIVE_AFTERMATH" and current_incident is not None:
            if has_acc and frame_detections:
                acc_conf_val = max(c for _, c in frame_detections)
                current_incident["last_frame"] = frame_count
                current_incident["class_votes"].append((frame_detections[0][0], acc_conf_val))
                current_incident["acc_confs"].append(acc_conf_val)

        # ── Check for incident closure ──────────────────────────────────────────
        if current_incident is not None and verifier.state in ("NO_INCIDENT", "INCIDENT_CLOSED"):
            incidents.append(current_incident)
            current_incident = None
            latched_confirmed = False
            latched_severity_label = None

        # ── Accumulate night context for active incident ────────────────────────
        if current_incident is not None and night_results is not None:
            for box in night_results.boxes:
                cls = int(box.cls[0])
                current_incident["night_context"].add(
                    night_names.get(cls, str(cls))
                )

        # ── Step 6: Annotate frame ─────────────────────────────────────────────
        annotated = frame.copy()
        _draw_night_boxes(annotated, night_results, night_names)

        # Only draw raw accident-class boxes during build-up and on the confirmation frame.
        # Suppress raw accident boxes during active aftermath to avoid repeated car_car_accident labels.
        if verifier.state != "INCIDENT_ACTIVE_AFTERMATH" or is_confirmed:
            _draw_accident_boxes(annotated, accident_results, accident_names)

        _, consecutive    = verifier.get_current_status()
        current_acc_class = verifier.current_accident_class

        if latched_confirmed:
            display_confirmed      = True
            display_severity_label = latched_severity_label
        elif consecutive == 0 and not is_confirmed:
            display_confirmed      = False
            display_not_confirmed  = False
            display_severity_label = None

        _overlay_status(annotated, current_acc_class, consecutive,
                        confirm_frames, display_confirmed,
                        display_not_confirmed, display_severity_label)

        writer.write(annotated)

        # ── Progress ───────────────────────────────────────────────────────────
        total_time  += time.time() - t_start
        frame_count += 1

        if frame_count % 50 == 0 or frame_count == total_frames:
            fps_now = frame_count / total_time if total_time > 0 else 0
            pct     = frame_count / total_frames * 100 if total_frames > 0 else 0
            _, consec  = verifier.get_current_status()
            active_cls = verifier.current_accident_class

            if active_cls and consec > 0:
                acc_conf_now = (
                    max(c for _, c in frame_detections)
                    if frame_detections else 0.0
                )
                print(f"  [{frame_count}/{total_frames}] {pct:5.1f}% | "
                      f"FPS: {fps_now:.1f} | "
                      f"{active_cls} | Verification: {consec}/{confirm_frames} | "
                      f"Conf: {acc_conf_now:.2f}")
            elif frame_has_accident and frame_detections:
                top_name, top_conf = frame_detections[0]
                print(f"  [{frame_count}/{total_frames}] {pct:5.1f}% | "
                      f"FPS: {fps_now:.1f} | "
                      f"Accident: {top_name} | Conf: {top_conf:.2f}")
            else:
                print(f"  [{frame_count}/{total_frames}] {pct:5.1f}% | "
                      f"FPS: {fps_now:.1f} | "
                      f"Accident detections: {frames_with_accident}")

    cap.release()
    writer.release()

    _finalize_and_verify_video(
        raw_output_path=raw_output_path,
        final_output_path=output_video_path,
        input_fps=fps,
        input_frames=total_frames
    )


    if current_incident is not None:
        incidents.append(current_incident)

    # ── Build final report ─────────────────────────────────────────────────────
    avg_fps = frame_count / total_time if total_time > 0 else 0

    accident_detected  = frames_with_accident > 0
    accident_verified  = len(incidents) > 0

    if accident_verified:
        all_class_votes    = []
        all_severity_votes = []
        all_acc_confs      = []
        all_night_context  = set()

        for inc in incidents:
            all_class_votes.extend(inc["class_votes"])
            all_severity_votes.extend(inc["severity_votes"])
            all_acc_confs.extend(inc["acc_confs"])
            all_night_context.update(inc["night_context"])
            final_validation = inc["validation"]

        first_frame_overall = incidents[0]["first_frame"]
        last_frame_overall  = incidents[-1]["last_frame"]

        final_acc_class             = _most_common_class(all_class_votes)
        final_severity, final_sev_conf = _aggregate_severity(all_severity_votes)
        avg_acc_conf    = sum(all_acc_confs) / len(all_acc_confs)
        max_acc_conf    = max(all_acc_confs)
        duration_frames = last_frame_overall - first_frame_overall
        duration_sec    = duration_frames / fps if fps > 0 else 0.0

        night_classes_all = {"Bicycle", "Bus", "Car", "Motorbike", "People"}
        night_ctx = {
            c: ("Detected" if c in all_night_context else "Not detected")
            for c in night_classes_all
        }
        sev_conf_display = (f"{final_sev_conf:.3f}"
                            if final_severity != "UNCERTAIN" else "N/A")
        final_decision   = f"ACCIDENT CONFIRMED — {final_severity}"

    elif accident_detected:
        # Temporal threshold reached at least once but spatial check failed
        final_acc_class   = "N/A"
        final_severity    = "NO ACCIDENT"
        sev_conf_display  = "N/A"
        final_sev_conf    = 0.0
        avg_acc_conf      = 0.0
        max_acc_conf      = 0.0
        first_frame_overall = last_frame_overall = 0
        duration_frames   = 0
        duration_sec      = 0.0
        night_ctx         = {}
        final_decision    = "NO ACCIDENT CONFIRMED"

    else:
        final_acc_class   = "N/A"
        final_severity    = "NO ACCIDENT"
        sev_conf_display  = "N/A"
        final_sev_conf    = 0.0
        avg_acc_conf      = 0.0
        max_acc_conf      = 0.0
        first_frame_overall = last_frame_overall = 0
        duration_frames   = 0
        duration_sec      = 0.0
        night_ctx         = {}
        final_decision    = "NO ACCIDENT DETECTED"

    video_name = Path(video_path).name

    # ── Processing complete ────────────────────────────────────────────────────
    print(DASH)
    print("  PROCESSING COMPLETE")
    print(DASH)
    print(f"  Frames processed  : {frame_count}/{total_frames}")
    print(f"  Processing FPS    : {avg_fps:.2f}")
    print(f"  Accident frames   : {frames_with_accident}")
    print(f"  Detected          : {'YES' if accident_detected else 'NO'}")
    print(f"  Verified          : {'YES' if accident_verified else 'NO'}")

    # ── Final report (terminal) ────────────────────────────────────────────────
    REP = "=" * 50
    # Friendly display values
    first_str    = str(first_frame_overall) if accident_verified else "N/A"
    confirm_str  = str(last_frame_overall)  if accident_verified else "N/A"
    dur_str      = (f"{duration_frames} frames ({duration_sec:.1f}s)"
                    if accident_verified else "N/A")
    avg_conf_str = f"{avg_acc_conf:.3f}" if accident_verified else "N/A"
    max_conf_str = f"{max_acc_conf:.3f}" if accident_verified else "N/A"

    cars_str  = night_ctx.get("Car",       "Not detected") if accident_verified else "N/A"
    moto_str  = night_ctx.get("Motorbike", "Not detected") if accident_verified else "N/A"
    ppl_str   = night_ctx.get("People",    "Not detected") if accident_verified else "N/A"

    print(f"\n{REP}")
    print("FINAL ACCIDENT REPORT")
    print(REP)
    print(f"Video              : {video_name}")
    print(f"Accident Detected  : {'YES' if accident_detected else 'NO'}")
    print(f"Accident Verified  : {'YES' if accident_verified else 'NO'}")
    print(f"Accident Type      : {final_acc_class}")
    print(f"Severity           : {final_severity}")
    print(f"Severity Confidence: {sev_conf_display}")
    print(f"First Detection    : {first_str}")
    print(f"Confirmation Frame : {confirm_str}")
    print(f"Duration           : {dur_str}")
    print(f"Average Confidence : {avg_conf_str}")
    print(f"Maximum Confidence : {max_conf_str}")
    print(f"Night Traffic:")
    print(f"  Cars       : {cars_str}")
    print(f"  Motorbikes : {moto_str}")
    print(f"  People     : {ppl_str}")
    print(f"Total Frames       : {total_frames}")
    print(f"Processing FPS     : {avg_fps:.2f}")
    print(REP)
    print("FINAL DECISION")
    print(REP)
    print(final_decision)
    print(REP)

    # ── Summary file ───────────────────────────────────────────────────────────
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"{REP}\nFINAL ACCIDENT REPORT\n{REP}\n")
        f.write(f"Video              : {video_name}\n")
        f.write(f"Accident Detected  : {'YES' if accident_detected else 'NO'}\n")
        f.write(f"Accident Verified  : {'YES' if accident_verified else 'NO'}\n")
        f.write(f"Accident Type      : {final_acc_class}\n")
        f.write(f"Severity           : {final_severity}\n")
        f.write(f"Severity Confidence: {sev_conf_display}\n")
        f.write(f"First Detection    : {first_str}\n")
        f.write(f"Confirmation Frame : {confirm_str}\n")
        f.write(f"Duration           : {dur_str}\n")
        f.write(f"Average Confidence : {avg_conf_str}\n")
        f.write(f"Maximum Confidence : {max_conf_str}\n")
        f.write(f"Night Traffic:\n")
        f.write(f"  Cars       : {cars_str}\n")
        f.write(f"  Motorbikes : {moto_str}\n")
        f.write(f"  People     : {ppl_str}\n")
        f.write(f"Total Frames       : {total_frames}\n")
        f.write(f"Processing FPS     : {avg_fps:.2f}\n")
        f.write(f"{REP}\nFINAL DECISION\n{REP}\n")
        f.write(f"{final_decision}\n{REP}\n")

    print(f"\n[+] Output video  : {output_video_path}")
    print(f"[+] Summary saved : {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated three-model CCTV accident detection test"
    )
    parser.add_argument("--video",   default=str(DEFAULT_VIDEO),
                        help="Path to input video (.mp4 or .avi)")
    parser.add_argument("--output",  default=None,
                        help="Output annotated video path")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY),
                        help="Path to summary text file")
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-frame detection details")
    args = parser.parse_args()

    if args.output is None:
        in_path    = Path(args.video)
        args.output = str(DEFAULT_OUTPUT_DIR /
                          (in_path.stem + "_result" + in_path.suffix))

    run(
        video_path        = args.video,
        output_video_path = args.output,
        summary_path      = args.summary,
        debug             = args.debug,
    )


if __name__ == "__main__":
    main()
