"""
Video input diagnostic for 5.mp4.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT_DIR / "runs" / "detect" / "results" / "video_test" / "5.mp4"

def main():
    print(f"[*] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    if not cap.isOpened():
        print("[!] FAILED to open video")
        return
    
    print("[+] Video opened successfully")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"\nVideo Properties:")
    print(f"  Codec: {codec_str} (0x{codec:08X})")
    print(f"  FPS: {fps}")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print(f"  Total frames: {total_frames}")
    
    # Extract frames
    output_dir = ROOT_DIR / "results" / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_to_extract = [
        ("first", 0),
        ("middle", total_frames // 2),
        ("final", total_frames - 1)
    ]
    
    extracted_paths = []
    
    print(f"\n[*] Extracting frames...")
    
    for name, frame_num in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"[!] FAILED to read {name} frame (frame {frame_num})")
            continue
        
        print(f"[+] Successfully read {name} frame (frame {frame_num})")
        
        # Frame properties
        shape = frame.shape
        pixel_min = frame.min()
        pixel_max = frame.max()
        pixel_mean = frame.mean()
        
        print(f"  Shape: {shape}")
        print(f"  Pixel range: {pixel_min} to {pixel_max}")
        print(f"  Pixel mean: {pixel_mean:.2f}")
        
        # Check if blank/corrupted
        is_blank = pixel_max == pixel_min
        is_black = pixel_max == 0
        
        if is_blank:
            print(f"  WARNING: Frame appears BLANK (all pixels same value)")
        elif is_black:
            print(f"  WARNING: Frame appears BLACK (all pixels zero)")
        else:
            print(f"  Frame appears to have content")
        
        # Save frame
        output_path = output_dir / f"5_{name}_frame.jpg"
        cv2.imwrite(str(output_path), frame)
        extracted_paths.append(output_path)
        print(f"  Saved to: {output_path}")
    
    cap.release()
    
    print("\n" + "=" * 80)
    print("VIDEO INPUT DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nExtracted frames:")
    for path in extracted_paths:
        print(f"  {path}")
    
    print("\nNOTE: Please visually inspect the extracted images to verify they contain")
    print("the expected road/vehicles/accident scene.")

if __name__ == "__main__":
    main()
